import torch
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
import numpy as np

class SISDistortionGraphExplainer(torch.nn.Module):

    def __init__(self, model, log=True, greedy=True):
        super(SISDistortionGraphExplainer, self).__init__()
        self.model = model
        self.log = log

        self.greedy = greedy
        if self.greedy:
            self.greediness = 10
            self.sorted_possible_nodes = []
            self.sorted_possible_features = []

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def distortion(self, node_idx=None, full_feature_matrix=None, computation_graph_feature_matrix=None,
                   edge_index=None, node_mask=None, feature_mask=None, predicted_label=None, samples=100,
                   random_seed=12345):
        if node_idx is None:
            node_idx = self.node_idx

        if full_feature_matrix is None:
            full_feature_matrix = self.full_feature_matrix

        if computation_graph_feature_matrix is None:
            computation_graph_feature_matrix = self.computation_graph_feature_matrix

        if edge_index is None:
            edge_index = self.computation_graph_edge_index

        if node_mask is None:
            node_mask = self.selected_nodes

        if feature_mask is None:
            feature_mask = self.selected_features

        if predicted_label is None:
            predicted_label = self.predicted_label

        (num_nodes, num_features) = full_feature_matrix.size()

        num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

        # retrieve complete mask as matrix
        mask = node_mask.T.matmul(feature_mask)

        correct = 0

        rng = torch.Generator()
        rng.manual_seed(random_seed)
        random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                       generator=rng)
        for i in range(samples):
            random_features = torch.gather(full_feature_matrix,
                                           dim=0,
                                           index=random_indices[i, :, :])

            randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

            log_logits = self.model(x=randomized_features, edge_index=edge_index)
            distorted_labels = log_logits.argmax(dim=-1)

            if distorted_labels[node_idx] == predicted_label:
                correct += 1

        return correct / samples

    def argmax_distortion_general(self,
                                  previous_distortion,
                                  possible_elements,
                                  selected_elements,
                                  initialization=False,
                                  **distortion_kwargs,
                                  ):
        if self.greedy:
            # determine if node or features
            if selected_elements is not self.selected_nodes and selected_elements is not self.selected_features:
                raise Exception("Neither features nor nodes selected")
            if initialization:
                best_element, best_distortion_improve, raw_sorted_elements = self.argmax_distortion_general_full(
                    previous_distortion,
                    possible_elements,
                    selected_elements,
                    save_all_pairs=True,
                    **distortion_kwargs,
                )

                if selected_elements is self.selected_nodes:
                    self.sorted_possible_nodes = sorted(raw_sorted_elements, key=lambda x: x[1], reverse=True)
                else:
                    self.sorted_possible_features = sorted(raw_sorted_elements, key=lambda x: x[1], reverse=True)

                return best_element, best_distortion_improve

            else:
                if selected_elements is self.selected_nodes:
                    sorted_elements = self.sorted_possible_nodes
                else:
                    sorted_elements = self.sorted_possible_features

                restricted_possible_elements = torch.zeros_like(possible_elements)

                counter = 0
                for index, initial_distortion_improve in sorted_elements:
                    if possible_elements[0, index] == 1 and selected_elements[0, index] == 0:
                        counter += 1
                        restricted_possible_elements[0, index] = 1
                        # possible alternative based on initial distortion improve
                        if counter == self.greediness:
                            break

                    else:
                        # think about removing those elements
                        pass

                # add selected elements to possible elements to avoid -1 in the calculation of remaining elements
                restricted_possible_elements += selected_elements

                return self.argmax_distortion_general_full(
                    previous_distortion,
                    restricted_possible_elements,
                    selected_elements,
                    **distortion_kwargs,
                )

        else:
            return self.argmax_distortion_general_full(
                previous_distortion,
                possible_elements,
                selected_elements,
                **distortion_kwargs,
            )

    def argmax_distortion_general_full(self,
                                       previous_distortion,
                                       possible_elements,
                                       selected_elements,
                                       save_all_pairs=False,
                                       **distortion_kwargs,
                                       ):
        best_element = None
        best_distortion_improve = -1000

        remaining_nodes_to_select = possible_elements - selected_elements
        num_remaining = remaining_nodes_to_select.sum()

        # if no node left break
        if num_remaining == 0:
            return best_element, best_distortion_improve

        if self.log:  # pragma: no cover
            pbar = tqdm(total=int(num_remaining), position=0)
            pbar.set_description(f'Argmax {best_element}, {best_distortion_improve}')

        all_calculated_pairs = []

        i = 0
        while num_remaining > 0:
            if selected_elements[0, i] == 0 and possible_elements[0, i] == 1:
                num_remaining -= 1

                selected_elements[0, i] = 1

                distortion_improve = self.distortion(**distortion_kwargs) \
                                     - previous_distortion

                selected_elements[0, i] = 0

                if save_all_pairs:
                    all_calculated_pairs.append((i, distortion_improve))

                if distortion_improve > best_distortion_improve:
                    best_element = i
                    best_distortion_improve = distortion_improve
                    if self.log:  # pragma: no cover
                        pbar.set_description(f'Argmax {best_element}, {best_distortion_improve}')

                if self.log:  # pragma: no cover
                    pbar.update(1)
            i += 1

        if self.log:  # pragma: no cover
            pbar.close()
        if save_all_pairs:
            return best_element, best_distortion_improve, all_calculated_pairs
        else:
            return best_element, best_distortion_improve

    def _determine_minimal_set(self, initial_distortion, tau, possible_nodes, possible_features):
        current_distortion = initial_distortion
        executed_selections = [[None, None, current_distortion]]

        num_selected_nodes = 0
        num_selected_features = 0

        while current_distortion <= 1 - tau:

            if num_selected_nodes == num_selected_features == 0:
                best_node, improve_in_distortion_by_node = self.argmax_distortion_general(
                    current_distortion,
                    possible_nodes,
                    self.selected_nodes,
                    initialization=True,
                    feature_mask=possible_features,  # assume all features are selected
                )

                best_feature, improve_in_distortion_by_feature = self.argmax_distortion_general(
                    current_distortion,
                    possible_features,
                    self.selected_features,
                    initialization=True,
                    node_mask=possible_nodes,  # assume all nodes are selected
                )

            elif num_selected_features == 0:
                best_node, improve_in_distortion_by_node = None, -100

                best_feature, improve_in_distortion_by_feature = self.argmax_distortion_general(
                    current_distortion,
                    possible_features,
                    self.selected_features,
                )

            elif num_selected_nodes == 0:
                best_node, improve_in_distortion_by_node = self.argmax_distortion_general(
                    current_distortion,
                    possible_nodes,
                    self.selected_nodes,
                )

                best_feature, improve_in_distortion_by_feature = None, -100

            else:
                best_node, improve_in_distortion_by_node = self.argmax_distortion_general(
                    current_distortion,
                    possible_nodes,
                    self.selected_nodes,
                )

                best_feature, improve_in_distortion_by_feature = self.argmax_distortion_general(
                    current_distortion,
                    possible_features,
                    self.selected_features,
                )

            if best_node is None and best_feature is None:
                break

            if best_node is None:
                self.selected_features[0, best_feature] = 1
                num_selected_features += 1
                executed_selection = [None, best_feature]
            elif best_feature is None:
                self.selected_nodes[0, best_node] = 1
                num_selected_nodes += 1
                executed_selection = [best_node, None]
            elif improve_in_distortion_by_feature >= improve_in_distortion_by_node:
                # on equal improve prefer feature
                self.selected_features[0, best_feature] = 1
                num_selected_features += 1
                executed_selection = [None, best_feature]
            else:
                self.selected_nodes[0, best_node] = 1
                num_selected_nodes += 1
                executed_selection = [best_node, None]

            current_distortion = self.distortion()

            print(current_distortion)
            executed_selection.append(current_distortion)
            executed_selections.append(executed_selection)

            self.epoch += 1

            if self.log:  # pragma: no cover
                self.overall_progress_bar.update(1)

        return executed_selections

    def recursively_get_minimal_sets(self, initial_distortion, tau, possible_nodes, possible_features):

        print("Possible features", int(possible_features.sum()))
        print("Possible nodes", int(possible_nodes.sum()))

        # check maximal possible distortion with current possible nodes and features
        reachable_distortion = self.distortion(
            node_mask=possible_nodes,
            feature_mask=possible_features,
        )
        print("Maximal reachable distortion in this path", reachable_distortion)
        if reachable_distortion <= 1 - tau:
            return None

        executed_selections = self._determine_minimal_set(initial_distortion, tau, possible_nodes, possible_features)

        minimal_nodes_and_features_sets = [
            (self.selected_nodes.numpy(),
             self.selected_features.numpy(),
             executed_selections)
        ]

        return minimal_nodes_and_features_sets

        self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes))
        self.selected_features = torch.zeros((1, self.num_features))

        reduced_nodes = possible_nodes - minimal_nodes_and_features_sets[0][0]
        reduced_features = possible_features - minimal_nodes_and_features_sets[0][1]

        reduced_node_results = self.recursively_get_minimal_sets(
            initial_distortion,
            tau,
            reduced_nodes,
            possible_features,
        )
        if reduced_node_results is not None:
            minimal_nodes_and_features_sets.extend(reduced_node_results)

        self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes))
        self.selected_features = torch.zeros((1, self.num_features))

        reduced_feature_results = self.recursively_get_minimal_sets(
            initial_distortion,
            tau,
            possible_nodes,
            reduced_features,
        )
        if reduced_feature_results is not None:
            minimal_nodes_and_features_sets.extend(reduced_feature_results)

        return minimal_nodes_and_features_sets

    def explain_node(self, node_idx, full_feature_matrix, edge_index, tau=0.15):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()

        num_edges = edge_index.size(1)

        (num_nodes, self.num_features) = full_feature_matrix.size()

        self.node_idx = node_idx
        self.full_feature_matrix = full_feature_matrix

        # Only operate on a k-hop subgraph around `node_idx`.
        self.computation_graph_feature_matrix, self.computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
            self.__subgraph__(node_idx, full_feature_matrix, edge_index)

        self.num_computation_graph_nodes = self.computation_graph_feature_matrix.size(0)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(x=self.computation_graph_feature_matrix,
                                    edge_index=self.computation_graph_edge_index)
            predicted_labels = log_logits.argmax(dim=-1)

            self.predicted_label = predicted_labels[mapping]

            # self.__set_masks__(computation_graph_feature_matrix, edge_index)
            self.to(self.computation_graph_feature_matrix.device)

            if self.log:  # pragma: no cover
                self.overall_progress_bar = tqdm(total=int(self.num_computation_graph_nodes*self.num_features),
                                                 position=1)
                self.overall_progress_bar.set_description(f'Explain node {node_idx}')

            possible_nodes = torch.ones((1, self.num_computation_graph_nodes))
            possible_features = torch.ones((1, self.num_features))

            self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes))
            self.selected_features = torch.zeros((1, self.num_features))

            initial_distortion = self.distortion()

            self.epoch = 1
            minimal_nodes_and_features_sets = self.recursively_get_minimal_sets(initial_distortion,
                                                                                tau,
                                                                                possible_nodes,
                                                                                possible_features)

            if self.log:  # pragma: no cover
                self.overall_progress_bar.close()

        return minimal_nodes_and_features_sets


def save_minimal_nodes_and_features_sets(path_prefix, node, minimal_nodes_and_features_sets):
    path = path_prefix + "_node_" + str(node) + ".npz"

    numpy_dict = {
        "node": np.array(node),
        "number_of_sets": np.array(len(minimal_nodes_and_features_sets)),
    }

    features_label = "features_"
    nodes_label = "nodes_"
    selection_label = "selection_"

    for i, (selected_nodes, selected_features, executed_selections) in enumerate(minimal_nodes_and_features_sets):
        numpy_dict[nodes_label + str(i)] = selected_nodes
        numpy_dict[features_label + str(i)] = selected_features
        numpy_dict[selection_label + str(i)] = np.array(executed_selections)

    np.savez_compressed(path, **numpy_dict)


import torch
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import APPNP
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import logging
import time



class AbstractGraphExplainer(torch.nn.Module):

    def __init__(self, model, device, log=True, record_process_time=False):
        super(AbstractGraphExplainer, self).__init__()
        self.model = model
        self.log = log
        self.logger = logging.getLogger("explainer")
        self.device = device

        self.record_process_time = record_process_time

    @staticmethod
    def num_hops(model):
        num_hops = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                if isinstance(module, APPNP):
                    num_hops += module.K
                else:
                    num_hops += 1
        return num_hops

    def __num_hops__(self):
        return self.num_hops(self.model)

    @staticmethod
    def flow(model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __flow__(self):
        return self.flow(self.model)

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

        return subset, x, edge_index, mapping, edge_mask, kwargs

    def distortion(self, node_idx=None, full_feature_matrix=None, computation_graph_feature_matrix=None,
                   edge_index=None, node_mask=None, feature_mask=None, predicted_label=None, samples=None,
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

        if samples is None:
            samples = self.distortion_samples

        return distortion(self.model,
                          node_idx=node_idx,
                          full_feature_matrix=full_feature_matrix,
                          computation_graph_feature_matrix=computation_graph_feature_matrix,
                          edge_index=edge_index,
                          node_mask=node_mask,
                          feature_mask=feature_mask,
                          predicted_label=predicted_label,
                          samples=samples,
                          random_seed=random_seed,
                          device=self.device,
                          )


class Zorro(AbstractGraphExplainer):

    def __init__(self, model, device, log=True, greedy=True, record_process_time=False, add_noise=False, samples=100,
                 path_to_precomputed_distortions=None):
        super(Zorro, self).__init__(
            model=model,
            device=device,
            log=log,
            record_process_time=record_process_time,
        )
        self.distortion_samples = samples

        self.ensure_improvement = False

        self.add_noise = add_noise

        self.initial_node_improve = [np.nan]
        self.initial_feature_improve = [np.nan]

        self.greedy = greedy
        if self.greedy:
            self.greediness = 10
            self.sorted_possible_nodes = []
            self.sorted_possible_features = []

        self.path_to_precomputed_distortions = path_to_precomputed_distortions
        self.precomputed_distortion_info = {}

    def load_initial_distortion(self, node, neighbor_subset):
        saved_info = np.load(self.path_to_precomputed_distortions)

        nodes = list(saved_info["nodes"])
        subset = list(saved_info["subset"])
        mapping = saved_info["mapping"]
        initial_distortion = saved_info["initial_distortion"]
        feature_distortion = saved_info["feature_distortion"]
        node_distortion = saved_info["node_distortion"]

        if node not in nodes:
            raise ValueError("Node " + str(node) + "not found in precomputed distortions file "
                             + str(self.path_to_precomputed_distortions))

        position = nodes.index(node)

        best_feature = None
        best_feature_distortion_improve = -1000
        raw_unsorted_features = []
        for i in range(feature_distortion.shape[0]):
            distortion_improve = feature_distortion[i, position] - initial_distortion[position]
            raw_unsorted_features.append((i, distortion_improve))
            if distortion_improve > best_feature_distortion_improve:
                best_feature_distortion_improve = distortion_improve
                best_feature = i

        best_node = None
        best_node_distortion_improve = -1000
        raw_unsorted_nodes = []
        for i, neighbor in enumerate(neighbor_subset):
            if subset.index(neighbor) == -1:
                raise ValueError("Neighbor " + str(neighbor) + "not found in precomputed neighbors " + str(subset))
            distortion_improve = node_distortion[subset.index(neighbor), position] - initial_distortion[position]
            raw_unsorted_nodes.append((i, distortion_improve))
            if distortion_improve > best_node_distortion_improve:
                best_node_distortion_improve = distortion_improve
                best_node = i

        # save infos in dict
        self.precomputed_distortion_info["best_node"] = best_node
        self.precomputed_distortion_info["best_node_distortion_improve"] = best_node_distortion_improve
        self.precomputed_distortion_info["raw_unsorted_nodes"] = raw_unsorted_nodes
        self.precomputed_distortion_info["best_feature"] = best_feature
        self.precomputed_distortion_info["best_feature_distortion_improve"] = best_feature_distortion_improve
        self.precomputed_distortion_info["raw_unsorted_features"] = raw_unsorted_features
        self.logger.debug("Successfully loaded precomputed information")

    def argmax_distortion_general(self,
                                  previous_distortion,
                                  possible_elements,
                                  selected_elements,
                                  initialization=False,
                                  save_initial_improve=False,
                                  **distortion_kwargs,
                                  ):
        if self.greedy:
            # determine if node or features
            if selected_elements is not self.selected_nodes and selected_elements is not self.selected_features:
                raise Exception("Neither features nor nodes selected")
            if initialization:
                if self.epoch == 1 and self.precomputed_distortion_info:
                    if selected_elements is self.selected_nodes:
                        best_element = self.precomputed_distortion_info["best_node"]
                        best_distortion_improve = self.precomputed_distortion_info["best_node_distortion_improve"]
                        raw_sorted_elements = self.precomputed_distortion_info["raw_unsorted_nodes"]
                        self.logger.debug("Used precomputed node info")
                    else:
                        best_element = self.precomputed_distortion_info["best_feature"]
                        best_distortion_improve = self.precomputed_distortion_info["best_feature_distortion_improve"]
                        raw_sorted_elements = self.precomputed_distortion_info["raw_unsorted_features"]
                        self.logger.debug("Used precomputed feature info")

                else:
                    best_element, best_distortion_improve, raw_sorted_elements = self.argmax_distortion_general_full(
                        previous_distortion,
                        possible_elements,
                        selected_elements,
                        save_all_pairs=True,
                        **distortion_kwargs,
                    )

                if selected_elements is self.selected_nodes:
                    self.sorted_possible_nodes = sorted(raw_sorted_elements, key=lambda x: x[1], reverse=True)
                    if save_initial_improve:
                        self.initial_node_improve = raw_sorted_elements
                else:
                    self.sorted_possible_features = sorted(raw_sorted_elements, key=lambda x: x[1], reverse=True)
                    if save_initial_improve:
                        self.initial_feature_improve = raw_sorted_elements

                return best_element, best_distortion_improve

            else:
                if selected_elements is self.selected_nodes:
                    sorted_elements = self.sorted_possible_nodes
                else:
                    sorted_elements = self.sorted_possible_features

                restricted_possible_elements = torch.zeros_like(possible_elements, device=self.device)

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

                best_element, best_distortion_improve = self.argmax_distortion_general_full(
                    previous_distortion,
                    restricted_possible_elements,
                    selected_elements,
                    **distortion_kwargs,
                )

                return best_element, best_distortion_improve

        elif save_initial_improve:
            best_element, best_distortion_improve, raw_sorted_elements = self.argmax_distortion_general_full(
                previous_distortion,
                possible_elements,
                selected_elements,
                save_all_pairs=True,
                **distortion_kwargs,
            )

            if selected_elements is self.selected_nodes:
                self.initial_node_improve = raw_sorted_elements
            else:
                self.initial_feature_improve = raw_sorted_elements

            return best_element, best_distortion_improve
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

    def _determine_minimal_set(self, initial_distortion, tau, possible_nodes, possible_features,
                               save_initial_improve=False):
        current_distortion = initial_distortion
        if self.record_process_time:
            last_time = time.time()
            executed_selections = [[np.nan, np.nan, current_distortion, 0]]
        else:
            last_time = 0
            executed_selections = [[np.nan, np.nan, current_distortion]]

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
                    save_initial_improve=save_initial_improve,
                )

                best_feature, improve_in_distortion_by_feature = self.argmax_distortion_general(
                    current_distortion,
                    possible_features,
                    self.selected_features,
                    initialization=True,
                    node_mask=possible_nodes,  # assume all nodes are selected
                    save_initial_improve=save_initial_improve,
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

            if self.ensure_improvement and \
                    improve_in_distortion_by_node < .00000001 and improve_in_distortion_by_feature < .00000001:
                pass

            if best_node is None and best_feature is None:
                break

            if best_node is None:
                self.selected_features[0, best_feature] = 1
                num_selected_features += 1
                executed_selection = [np.nan, best_feature]
            elif best_feature is None:
                self.selected_nodes[0, best_node] = 1
                num_selected_nodes += 1
                executed_selection = [best_node, np.nan]
            elif improve_in_distortion_by_feature >= improve_in_distortion_by_node:
                # on equal improve prefer feature
                self.selected_features[0, best_feature] = 1
                num_selected_features += 1
                executed_selection = [np.nan, best_feature]
            else:
                self.selected_nodes[0, best_node] = 1
                num_selected_nodes += 1
                executed_selection = [best_node, np.nan]

            current_distortion = self.distortion()

            print(current_distortion)
            executed_selection.append(current_distortion)

            if self.record_process_time:
                executed_selection.append(time.time() - last_time)
                last_time = time.time()

            executed_selections.append(executed_selection)

            self.epoch += 1

            if self.log:  # pragma: no cover
                self.overall_progress_bar.update(1)

        return executed_selections

    def recursively_get_minimal_sets(self, initial_distortion, tau, possible_nodes, possible_features,
                                     recursion_depth=np.inf, save_initial_improve=False):

        self.logger.debug("  Possible features " + str(int(possible_features.sum())))
        self.logger.debug("  Possible nodes " + str(int(possible_nodes.sum())))

        # check maximal possible distortion with current possible nodes and features
        reachable_distortion = self.distortion(
            node_mask=possible_nodes,
            feature_mask=possible_features,
        )
        self.logger.debug("Maximal reachable distortion in this path " + str(reachable_distortion))
        if reachable_distortion <= 1 - tau:
            return None

        if recursion_depth == 0:
            return [(np.nan, np.nan, np.nan)]

        executed_selections = self._determine_minimal_set(initial_distortion, tau, possible_nodes, possible_features,
                                                          save_initial_improve=save_initial_improve)

        minimal_nodes_and_features_sets = [
            (self.selected_nodes.cpu().numpy(),
             self.selected_features.cpu().numpy(),
             executed_selections)
        ]

        self.logger.debug(" Explanation found")
        self.logger.debug(" Selected features " + str(int(minimal_nodes_and_features_sets[0][1].sum())))
        self.logger.debug(" Selected nodes " + str(int(minimal_nodes_and_features_sets[0][0].sum())))

        self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes), device=self.device)
        self.selected_features = torch.zeros((1, self.num_features), device=self.device)

        reduced_nodes = possible_nodes - torch.as_tensor(minimal_nodes_and_features_sets[0][0], device=self.device)
        reduced_features = possible_features - torch.as_tensor(minimal_nodes_and_features_sets[0][1],
                                                               device=self.device)

        reduced_node_results = self.recursively_get_minimal_sets(
            initial_distortion,
            tau,
            reduced_nodes,
            possible_features,
            recursion_depth=recursion_depth - 1,
            save_initial_improve=False,
        )
        if reduced_node_results is not None:
            minimal_nodes_and_features_sets.extend(reduced_node_results)

        self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes), device=self.device)
        self.selected_features = torch.zeros((1, self.num_features), device=self.device)

        reduced_feature_results = self.recursively_get_minimal_sets(
            initial_distortion,
            tau,
            possible_nodes,
            reduced_features,
            recursion_depth=recursion_depth - 1,
            save_initial_improve=False,
        )
        if reduced_feature_results is not None:
            minimal_nodes_and_features_sets.extend(reduced_feature_results)

        return minimal_nodes_and_features_sets

    def explain_node(self, node_idx, full_feature_matrix, edge_index, tau=0.15, recursion_depth=np.inf,
                     save_initial_improve=False):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        if save_initial_improve:
            self.initial_node_improve = [np.nan]
            self.initial_feature_improve = [np.nan]

        self.model.eval()

        if recursion_depth <= 0:
            self.logger.warning("Recursion depth not positve " + str(recursion_depth))
            raise ValueError("Recursion depth not positve " + str(recursion_depth))

        self.logger.info("------ Start explaining node " + str(node_idx))
        self.logger.debug("Distortion drop (tau): " + str(tau))
        self.logger.debug("Distortion samples: " + str(self.distortion_samples))
        self.logger.debug("Greedy variant: " + str(self.greedy))
        if self.greedy:
            self.logger.debug("Greediness: " + str(self.greediness))
            self.logger.debug("Ensure improvement: " + str(self.ensure_improvement))

        num_edges = edge_index.size(1)

        (num_nodes, self.num_features) = full_feature_matrix.size()

        self.full_feature_matrix = full_feature_matrix

        # Only operate on a k-hop subgraph around `node_idx`.
        neighbor_subset, self.computation_graph_feature_matrix, self.computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
            self.__subgraph__(node_idx, full_feature_matrix, edge_index)

        if self.add_noise:
            self.full_feature_matrix = torch.cat(
                [self.full_feature_matrix, torch.zeros_like(self.full_feature_matrix)],
                dim=0)

        self.node_idx = mapping

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
                self.overall_progress_bar = tqdm(total=int(self.num_computation_graph_nodes * self.num_features),
                                                 position=1)
                self.overall_progress_bar.set_description(f'Explain node {node_idx}')

            possible_nodes = torch.ones((1, self.num_computation_graph_nodes), device=self.device)
            possible_features = torch.ones((1, self.num_features), device=self.device)

            self.selected_nodes = torch.zeros((1, self.num_computation_graph_nodes), device=self.device)
            self.selected_features = torch.zeros((1, self.num_features), device=self.device)

            initial_distortion = self.distortion()

            # safe the unmasked distortion
            self.logger.debug("Initial distortion without any mask: " + str(initial_distortion))

            if initial_distortion >= 1 - tau:
                # no mask needed, global distribution enough, see node 1861 in cora_GINConv
                self.logger.info("------ Finished explaining node " + str(node_idx))
                self.logger.debug("# Explanations: Select any nodes and features")
                if save_initial_improve:
                    return [
                               (self.selected_nodes.cpu().numpy(),
                                self.selected_features.cpu().numpy(),
                                [[np.nan, np.nan, initial_distortion], ]
                                )
                           ], None, None
                else:
                    return [
                        (self.selected_nodes.cpu().numpy(),
                         self.selected_features.cpu().numpy(),
                         [[np.nan, np.nan, initial_distortion], ]
                         )
                    ]
            else:

                # if available load precomputed distortions
                if self.path_to_precomputed_distortions is not None:
                    self.load_initial_distortion(node_idx, neighbor_subset)

                self.epoch = 1
                minimal_nodes_and_features_sets = self.recursively_get_minimal_sets(
                    initial_distortion,
                    tau,
                    possible_nodes,
                    possible_features,
                    recursion_depth=recursion_depth,
                    save_initial_improve=save_initial_improve,
                )

            if self.log:  # pragma: no cover
                self.overall_progress_bar.close()

        self.logger.info("------ Finished explaining node " + str(node_idx))
        self.logger.debug("# Explanations: " + str(len(minimal_nodes_and_features_sets)))

        if save_initial_improve:
            return minimal_nodes_and_features_sets, self.initial_node_improve, self.initial_feature_improve
        else:
            return minimal_nodes_and_features_sets


class SoftZorro(AbstractGraphExplainer):
    coeffs = {
        'fidelity': 1,
        'node_size': 0.01,
        'node_ent': 0.1,
        'feature_size': 0.01,
        'feature_ent': 0.1,
    }

    def __init__(self, model, device, log=True, record_process_time=False, samples=100, learning_rate=0.01):
        super(SoftZorro, self).__init__(
            model=model,
            device=device,
            log=log,
            record_process_time=record_process_time,
        )
        self.distortion_samples = samples
        self.learning_rate = learning_rate

    def loss(self, node, node_mask, feature_mask, full_feature_matrix, computation_graph_feature_matrix,
             computation_graph_edge_index, predicted_label, return_soft_distortion=False):
        loss = -distortion(self.model, node,
                           node_mask=node_mask,
                           feature_mask=feature_mask,
                           full_feature_matrix=full_feature_matrix,
                           computation_graph_feature_matrix=computation_graph_feature_matrix,
                           edge_index=computation_graph_edge_index,
                           samples=100,
                           predicted_label=predicted_label,
                           random_seed=None,
                           soft_distortion=True,
                           device=self.device)

        EPS = 1e-15

        if return_soft_distortion:
            soft_distortion = loss.clone().detach().cpu().numpy()
        # weight of soft fidelity
        loss = self.coeffs["fidelity"] * loss

        m = node_mask
        loss = loss + self.coeffs["node_size"] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs["node_ent"] * ent.mean()

        m = feature_mask
        loss = loss + self.coeffs["feature_size"] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs["feature_ent"] * ent.mean()

        if return_soft_distortion:
            return loss, soft_distortion
        else:
            return loss

    def explain_node(self, node_idx, full_feature_matrix, edge_index):
        (num_nodes, num_features) = full_feature_matrix.size()

        # Only operate on a k-hop subgraph around `node_idx`.
        neighbor_subset, computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, full_feature_matrix, edge_index)

        self.model.eval()
        log_logits = self.model(x=computation_graph_feature_matrix,
                                edge_index=computation_graph_edge_index)
        predicted_labels = log_logits.argmax(dim=-1)

        predicted_label = predicted_labels[mapping]

        num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
        node_mask = torch.rand((1, num_computation_graph_nodes), device=self.device, requires_grad=True)
        feature_mask = torch.rand((1, num_features), device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([node_mask, feature_mask], lr=self.learning_rate)

        self.logger.info("------ Start explaining node " + str(node_idx))
        loss, soft_distortion = self.loss(mapping, node_mask, feature_mask, full_feature_matrix,
                                          computation_graph_feature_matrix,
                                          computation_graph_edge_index, predicted_label,
                                          return_soft_distortion=True)
        self.logger.debug("Initial distortion: " + str(-soft_distortion[0]))
        self.logger.debug("Initial Loss: " + str(loss.detach().cpu().numpy()[0]))

        execution_time = time.time()

        epochs = 200
        for i in range(epochs):
            if epochs > i > 0 and i % 25 == 0:
                loss, soft_distortion = self.loss(mapping, node_mask, feature_mask, full_feature_matrix,
                                                  computation_graph_feature_matrix,
                                                  computation_graph_edge_index, predicted_label,
                                                  return_soft_distortion=True)
            else:
                loss = self.loss(mapping, node_mask, feature_mask, full_feature_matrix,
                                 computation_graph_feature_matrix,
                                 computation_graph_edge_index, predicted_label)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                node_mask.clamp_(min=0, max=1)
                feature_mask.clamp_(min=0, max=1)

            if epochs > i > 0 and i % 25 == 0:
                self.logger.debug("Epoch: " + str(i))
                self.logger.debug("Distortion: " + str(-soft_distortion[0]))
                self.logger.debug("Loss: " + str(loss.detach().cpu().numpy()[0]))

        execution_time = time.time() - execution_time

        self.logger.info("------ Finished explaining node " + str(node_idx))
        loss, soft_distortion = self.loss(mapping, node_mask, feature_mask, full_feature_matrix,
                                          computation_graph_feature_matrix,
                                          computation_graph_edge_index, predicted_label,
                                          return_soft_distortion=True)
        self.logger.debug("Final distortion: " + str(-soft_distortion[0]))
        self.logger.debug("Final Loss: " + str(loss.detach().cpu().numpy()[0]))

        numpy_node_mask = node_mask.clone().detach().cpu().numpy()
        numpy_feature_mask = feature_mask.clone().detach().cpu().numpy()
        self.logger.debug("Possible nodes: " + str((numpy_node_mask >= 0).sum()))
        self.logger.debug("Non zero nodes: " + str((numpy_node_mask > 0).sum()))
        self.logger.debug("Non zero features: " + str((numpy_feature_mask > 0).sum()))

        if self.record_process_time:
            return numpy_node_mask, numpy_feature_mask, execution_time
        else:
            return numpy_node_mask, numpy_feature_mask


def save_soft_mask(save_path, node, node_mask, feature_mask, execution_time=np.inf):
    path = save_path

    numpy_dict = {
        "node": np.array(node),
        "node_mask": node_mask,
        "feature_mask": feature_mask,
        "execution_time": np.array(execution_time)
    }
    np.savez_compressed(path, **numpy_dict)


def load_soft_mask(path_prefix, node):
    path = path_prefix + "_node_" + str(node) + ".npz"

    save = np.load(path)
    node_mask = save["node_mask"]
    feature_mask = save["feature_mask"]
    execution_time = save["execution_time"]
    if execution_time is np.inf:
        return node_mask, feature_mask
    else:
        return node_mask, feature_mask, float(execution_time)


def save_minimal_nodes_and_features_sets(save_path, node, minimal_nodes_and_features_sets,
                                         initial_node_improve=None, initial_feature_improve=None):
    path = save_path

    if minimal_nodes_and_features_sets is None:
        numpy_dict = {
            "node": np.array(node),
            "number_of_sets": np.array(0),
        }

    else:

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

    if initial_node_improve is not None:
        numpy_dict["initial_node_improve"] = np.array(initial_node_improve)

    if initial_feature_improve is not None:
        numpy_dict["initial_feature_improve"] = np.array(initial_feature_improve)

    np.savez_compressed(path, **numpy_dict)


def load_minimal_nodes_and_features_sets(path_prefix, node, check_for_initial_improves=False):
    path = path_prefix + "_node_" + str(node) + ".npz"

    save = np.load(path, allow_pickle=False)

    saved_node = save["node"]
    if saved_node != node:
        raise ValueError("Other node then specified", saved_node, node)
    number_of_sets = save["number_of_sets"]

    minimal_nodes_and_features_sets = []

    if number_of_sets > 0:

        features_label = "features_"
        nodes_label = "nodes_"
        selection_label = "selection_"

        for i in range(number_of_sets):
            selected_nodes = save[nodes_label + str(i)]
            selected_features = save[features_label + str(i)]
            executed_selections = save[selection_label + str(i)]

            minimal_nodes_and_features_sets.append((selected_nodes, selected_features, executed_selections))

    if check_for_initial_improves:
        try:
            initial_node_improve = save["initial_node_improve"]
        except KeyError:
            initial_node_improve = None

        try:
            initial_feature_improve = save["initial_feature_improve"]
        except KeyError:
            initial_feature_improve = None

        return minimal_nodes_and_features_sets, initial_node_improve, initial_feature_improve
    else:
        return minimal_nodes_and_features_sets


def distortion(model, node_idx=None, full_feature_matrix=None, computation_graph_feature_matrix=None,
               edge_index=None, node_mask=None, feature_mask=None, predicted_label=None, samples=None,
               random_seed=12345, device="cpu", validity=False,
               soft_distortion=False, detailed_mask=None,
               ):
    # conditional_samples=True only works for int feature matrix!

    (num_nodes, num_features) = full_feature_matrix.size()

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix
    if detailed_mask is not None:
        mask = detailed_mask
    else:
        mask = node_mask.T.matmul(feature_mask)

    if validity:
        samples = 1
        full_feature_matrix = torch.zeros_like(full_feature_matrix)

    correct = 0.0

    rng = torch.Generator(device=device)
    if random_seed is not None:
        rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i, :, :])

        randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

        log_logits = model(x=randomized_features, edge_index=edge_index)
        if soft_distortion:
            correct += log_logits[node_idx].softmax(dim=-1).squeeze()[predicted_label]
        else:
            distorted_labels = log_logits.argmax(dim=-1)
            if distorted_labels[node_idx] == predicted_label:
                correct += 1
    return correct / samples


def multi_node_distortion(model,
                          nodes,
                          full_feature_matrix,
                          computation_graph_feature_matrix,
                          computation_graph_edge_index,
                          node_mask,
                          feature_mask,
                          predicted_labels,
                          samples=100,
                          random_seed=12345,
                          device="cpu",
                          ):
    (num_nodes, num_features) = full_feature_matrix.size()

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix
    mask = node_mask.T.matmul(feature_mask)

    correct = torch.zeros_like(predicted_labels)

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i, :, :])

        randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

        log_logits = model(x=randomized_features, edge_index=computation_graph_edge_index)
        distorted_labels = log_logits.argmax(dim=-1)

        correct[predicted_labels.eq(distorted_labels[nodes])] += 1

    return correct * (1 / float(samples))


def multi_node_precompute_full_distortion(model,
                                          nodes,
                                          full_feature_matrix,
                                          full_edge_index,
                                          save_path,
                                          samples=100,
                                          random_seed=12345,
                                          device="cpu",
                                          ):
    # get basic attributes: num_hops, flow
    num_hops = Zorro.num_hops(model)
    flow = Zorro.flow(model)

    (num_nodes, num_features) = full_feature_matrix.size()

    subset, computation_graph_edge_index, mapping, edge_mask = k_hop_subgraph(torch.tensor(nodes, device=device),
                                                                              num_hops,
                                                                              full_edge_index,
                                                                              relabel_nodes=True,
                                                                              num_nodes=num_nodes, flow=flow)

    computation_graph_feature_matrix = full_feature_matrix[subset]

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # calculate predicted labels
    log_logits = model(x=computation_graph_feature_matrix,
                       edge_index=computation_graph_edge_index)
    predicted_labels = log_logits.argmax(dim=-1)
    predicted_labels = predicted_labels[mapping]

    # calculate initial distortion
    node_mask = torch.zeros((1, num_nodes_computation_graph), device=device)
    feature_mask = torch.zeros((1, num_features), device=device)
    initial_distortion = multi_node_distortion(model,
                                               mapping,
                                               full_feature_matrix,
                                               computation_graph_feature_matrix,
                                               computation_graph_edge_index,
                                               node_mask,
                                               feature_mask,
                                               predicted_labels,
                                               samples=samples,
                                               random_seed=random_seed,
                                               device=device,
                                               )

    # calculate the improvement of features
    feature_distortion = torch.zeros((num_features, len(nodes)), device=device)
    node_mask = torch.ones_like(node_mask, device=device)
    for i in tqdm(range(num_features)):
        feature_mask[0, i] += 1

        feature_distortion[i] = multi_node_distortion(model,
                                                      mapping,
                                                      full_feature_matrix,
                                                      computation_graph_feature_matrix,
                                                      computation_graph_edge_index,
                                                      node_mask,
                                                      feature_mask,
                                                      predicted_labels,
                                                      samples=samples,
                                                      random_seed=random_seed,
                                                      device=device,
                                                      )
        feature_mask[0, i] -= 1

    # calculate the improvement of nodes
    node_distortion = torch.zeros((num_nodes_computation_graph, len(nodes)), device=device)

    feature_mask = torch.ones_like(feature_mask, device=device)
    node_mask = torch.zeros_like(node_mask, device=device)
    for i in tqdm(range(num_nodes_computation_graph)):
        node_mask[0, i] += 1

        node_distortion[i] = multi_node_distortion(model,
                                                   mapping,
                                                   full_feature_matrix,
                                                   computation_graph_feature_matrix,
                                                   computation_graph_edge_index,
                                                   node_mask,
                                                   feature_mask,
                                                   predicted_labels,
                                                   samples=samples,
                                                   random_seed=random_seed,
                                                   device=device,
                                                   )
        node_mask[0, i] -= 1

    np.savez_compressed(save_path,
                        **{
                            "nodes": nodes,
                            "subset": subset.cpu().numpy(),
                            "mapping": mapping.cpu().numpy(),
                            "initial_distortion": initial_distortion.cpu().numpy(),
                            "feature_distortion": feature_distortion.cpu().numpy(),
                            "node_distortion": node_distortion.cpu().numpy(),
                        }
                        )

    return subset, mapping, initial_distortion, feature_distortion, node_distortion

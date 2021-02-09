import torch
from gnn_explainer import GNNExplainer
from scipy.stats import entropy
import numpy as np
import pandas as pd


def binarize_tensor(tensor, number_of_ones):
    binary_tensor = torch.zeros_like(tensor)
    _, top_indices = torch.topk(tensor, number_of_ones, sorted=False)
    binary_tensor[top_indices] = 1

    return binary_tensor


SAVE_CACHE = ("path", {})


def get_gnn_distortion_with_cache(path_prefix, node, model, data,
                                  feature_mask, edge_mask=None, node_mask=None,
                                  feature_ones=None, edge_ones=None, node_ones=None,
                                  validity=False, single_save=False, skip_save=False,
                                  ):
    global SAVE_CACHE
    if single_save:
        save_path = str(path_prefix) + ".npz"
        key = str(node) + "_"
    else:
        save_path = str(path_prefix) + str(node) + ".npz"
        key = ""

    if feature_ones is None:
        key += "f_complete"
    else:
        key += "f_" + str(feature_ones)

    key += "_"

    if edge_mask is not None:
        if edge_ones is None:
            key += "e_complete"
        else:
            key += "e_" + str(edge_ones)

    if node_mask is not None:
        if node_ones is None:
            key += "n_complete"
        else:
            key += "n_" + str(node_ones)

    if validity:
        key += "_validity"

    if save_path != SAVE_CACHE[0]:
        SAVE_CACHE = (save_path, {})
        save = SAVE_CACHE[1]
        try:
            save_file = np.load(save_path)
            for saved_key in save_file:
                save[saved_key] = save_file[saved_key]
        except FileNotFoundError:
            pass
    save = SAVE_CACHE[1]

    if key not in save:
        if feature_ones is not None:
            feature_mask = binarize_tensor(feature_mask, feature_ones)

        if edge_ones is not None:
            edge_mask = binarize_tensor(edge_mask, edge_ones)

        if node_ones is not None:
            node_mask = binarize_tensor(node_mask, node_ones)

        if node_mask is not None:
            if len(node_mask.shape) == 1:
                node_mask = node_mask.unsqueeze(0)

        if len(feature_mask.shape) == 1:
            feature_mask = feature_mask.unsqueeze(0)

        save[key] = model.distortion(node, data.x, data.edge_index,
                                     feature_mask=feature_mask,
                                     edge_mask=edge_mask,
                                     node_mask=node_mask,
                                     validity=validity,
                                     )
        if not skip_save:
            np.savez_compressed(save_path, **save)

    return float(save[key])


def evaluate_explanations(explainer_name, model_name, dataset_name, model, data, explanations, global_results_directory,
                          epoch=-1, save_appendix=""):
    raw_explanation_info_softmax = {
        "explainer": [],
        "dataset": [],
        "model": [],
        "epoch": [],
        "node": [],
        "Fidelity": [],
        "Validity": [],
        "Entropy Feature Mask": [],
        "Entropy Edge Mask": [],
        "Entropy Node Mask": [],
        "Possible Nodes": [],
        "Possible Edges": [],
        "Possible Features": [],
        "SUM(Feature Mask)": [],
        "SUM(Edge Mask)": [],
        "SUM(Node Mask)": [],
    }

    # needed for the subgraph
    gnn_explainer = GNNExplainer(model, log=False)

    save_path = str(global_results_directory.joinpath(explainer_name + "_" + model_name + "_" + dataset_name))
    if epoch != -1:
        save_path += "_epoch_" + str(epoch)
    if save_appendix:
        save_path += save_appendix

    for counter, node in enumerate(explanations):
        feature_mask, edge_mask, node_mask = explanations[node]

        computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
            gnn_explainer.__subgraph__(node, data.x, data.edge_index)

        distortion = get_gnn_distortion_with_cache(save_path,
                                                   node,
                                                   gnn_explainer,
                                                   feature_mask=feature_mask,
                                                   edge_mask=edge_mask,
                                                   node_mask=node_mask,
                                                   data=data,
                                                   single_save=True,
                                                   skip_save=True,
                                                   )

        validity = get_gnn_distortion_with_cache(save_path,
                                                 node,
                                                 gnn_explainer,
                                                 feature_mask=feature_mask,
                                                 edge_mask=edge_mask,
                                                 node_mask=node_mask,
                                                 data=data,
                                                 validity=True,
                                                 single_save=True,
                                                 skip_save=(counter != len(explanations) - 1) and (counter % 25 != 0),
                                                 )

        raw_explanation_info_softmax["explainer"].append(explainer_name)
        raw_explanation_info_softmax["dataset"].append(dataset_name)
        raw_explanation_info_softmax["model"].append(model_name)
        raw_explanation_info_softmax["epoch"].append(epoch)
        raw_explanation_info_softmax["node"].append(node)
        raw_explanation_info_softmax["Fidelity"].append(distortion)
        raw_explanation_info_softmax["Validity"].append(validity)
        raw_explanation_info_softmax["SUM(Feature Mask)"].append(np.sum(feature_mask.numpy()))
        if np.abs(raw_explanation_info_softmax["SUM(Feature Mask)"][-1]) > 0.0001:
            raw_explanation_info_softmax["Entropy Feature Mask"].append(entropy(feature_mask.numpy().flatten()))
        else:
            raw_explanation_info_softmax["Entropy Feature Mask"].append(np.nan)

        if edge_mask is not None:
            raw_explanation_info_softmax["SUM(Edge Mask)"].append(np.sum(edge_mask.numpy()))
        else:
            raw_explanation_info_softmax["SUM(Edge Mask)"].append(0.0)
        if np.abs(raw_explanation_info_softmax["SUM(Edge Mask)"][-1]) > 0.0001:
            raw_explanation_info_softmax["Entropy Edge Mask"].append(entropy(edge_mask.numpy().flatten()))
        else:
            raw_explanation_info_softmax["Entropy Edge Mask"].append(np.nan)

        if node_mask is not None:
            raw_explanation_info_softmax["SUM(Node Mask)"].append(np.sum(node_mask.numpy()))
        else:
            raw_explanation_info_softmax["SUM(Node Mask)"].append(0.0)
        if np.abs(raw_explanation_info_softmax["SUM(Node Mask)"][-1]) > 0.0001:
            raw_explanation_info_softmax["Entropy Node Mask"].append(entropy(node_mask.numpy().flatten()))
        else:
            raw_explanation_info_softmax["Entropy Node Mask"].append(np.nan)

        num_nodes, num_features = computation_graph_feature_matrix.size()
        num_edges = computation_graph_edge_index.size(1)
        raw_explanation_info_softmax["Possible Nodes"].append(int(num_nodes))
        raw_explanation_info_softmax["Possible Features"].append(int(num_features))
        raw_explanation_info_softmax["Possible Edges"].append(int(num_edges))

    df_explanation_info = pd.DataFrame(data=raw_explanation_info_softmax)
    if not save_appendix:
        df_explanation_info.to_csv(save_path + "_info.csv")
    return df_explanation_info


def get_ground_truth_syn(node):
    # taken from https://github.com/vunhatminh/PGMExplainer/
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth


def evaluate_synthetic(explainer_name, model_name, dataset_name, model, data, explanations, global_results_directory,
                       epoch=-1):
    from torch_geometric.utils import k_hop_subgraph

    components_with_explained_node = []
    components_without_explained_node = []

    number_of_nodes_selected = []

    node_true_positive = []
    node_false_positive = []
    node_true_negative = []
    node_false_negative = []
    node_tpr = []
    node_precision = []
    node_accuracy = []

    # needed for the subgraph
    gnn_explainer = GNNExplainer(model, log=False)

    select_top_k_nodes = 10

    reduced_explanations = {}

    for node in explanations:
        subset, edge_index, _, _ = k_hop_subgraph(
            node, gnn_explainer.__num_hops__(), data.edge_index, relabel_nodes=False,
            num_nodes=None, flow=gnn_explainer.__flow__())

        subset = subset.numpy()

        node_ground_truth = set(get_ground_truth_syn(node))
        feature_mask, edge_mask, node_mask = explanations[node]

        if explainer_name == "GNNExplainer":
            # select top nodes based on edge mask
            top_edges_index = np.argpartition(edge_mask, -select_top_k_nodes)[-select_top_k_nodes:]
            # sort them descending (reason for the -edge_mask)
            top_edges_index = top_edges_index[np.argsort(-edge_mask[top_edges_index])]

            selected_nodes = set()
            for u, v in data.edge_index[:, top_edges_index].numpy().T:
                if len(selected_nodes) > 4:
                    break
                selected_nodes.add(u)
                selected_nodes.add(v)

            if len(selected_nodes) < 5:
                raise Exception("Not enough elements" + str(node))

            nodes_selected = set(selected_nodes)
            nodes_not_selected = set(subset).difference(nodes_selected)

        elif explainer_name in ["five_neighbors", "Grad", "GradInput"]:
            top_node_index = np.argpartition(node_mask, -5)[-5:]
            nodes_selected = set(subset[top_node_index])
            nodes_not_selected = set(subset).difference(nodes_selected)
        elif explainer_name == "edge_only":
            nodes_selected = set()
            nodes_not_selected = set(subset).difference(nodes_selected)
        elif explainer_name in ["Zorro", "Zorro_t_3"]:
            nodes_selected = set(subset[node_mask > 0])
            nodes_not_selected = set(subset).difference(nodes_selected)
        else:
            raise NotImplementedError("Not catched")

        number_of_nodes_selected.append(len(nodes_selected))

        # create top 5/6 node mask
        top_5_node_mask = torch.zeros(len(subset))
        list_subset = list(subset)
        for selected_neighbor in nodes_selected:
            top_5_node_mask[list_subset.index(selected_neighbor)] = 1

        reduced_explanations[node] = feature_mask, None, top_5_node_mask

        node_true_positive.append(len(nodes_selected.intersection(node_ground_truth)))
        node_false_positive.append(len(nodes_selected.difference(node_ground_truth)))
        node_true_negative.append(len(nodes_not_selected.difference(node_ground_truth)))
        node_false_negative.append(len(node_ground_truth.difference(nodes_selected)))

        node_tpr.append(node_true_positive[-1] / (node_true_positive[-1] + node_false_negative[-1]))
        if (node_true_positive[-1] + node_false_positive[-1]) > 0:
            node_precision.append(
                node_true_positive[-1] / (node_true_positive[-1] + node_false_positive[-1]))
        else:
            node_precision.append(np.nan)

        node_accuracy.append((node_true_positive[-1] + node_true_negative[-1]) / (
                node_true_positive[-1] + node_true_negative[-1]
                + node_false_positive[-1] + node_false_negative[-1]
        ))

    save_appendix = "top5"
    general_eval_of_top_5 = evaluate_explanations(
        explainer_name, model_name, dataset_name, model, data, reduced_explanations, global_results_directory,
        epoch=epoch, save_appendix=save_appendix)

    save_path = str(global_results_directory.joinpath(explainer_name + "_" + model_name + "_" + dataset_name))
    if epoch != -1:
        save_path += "_epoch_" + str(epoch)
    if save_appendix:
        save_path += save_appendix

    syn_details = pd.DataFrame(data={
        "#nodes": number_of_nodes_selected,
        "node_true_positive": node_true_positive,
        "node_false_positive": node_false_positive,
        "node_true_negative": node_true_negative,
        "node_false_negative": node_false_negative,
        "node_tpr": node_tpr,
        "node_precision": node_precision,
        "node_accuracy": node_accuracy}
    )

    full_evaluation = pd.concat([general_eval_of_top_5, syn_details], axis=1)
    full_evaluation.to_csv(save_path + "_info.csv")

    return full_evaluation

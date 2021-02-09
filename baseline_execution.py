import torch

from pathlib import Path
import numpy as np
import pandas as pd
import time

from models import load_model, load_dataset, GCNNet, GATNet, APPNP2Net, GINConvNet, GCN_syn2
from execution import MODEL_SAVE_NAMES

from gnn_explainer import GNNExplainer
from grad_explainer import grad_edge_explanation, grad_node_explanation, gradinput_node_explanation

from evaluation import evaluate_explanations, evaluate_synthetic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_gnn_explanation(path_prefix, node, feature_mask, edge_mask=None, node_mask=None):
    save_dict = {"node": np.array(node),
                 "feature_mask": np.array(feature_mask),
                 }
    if edge_mask is not None:
        save_dict["edge_mask"] = np.array(edge_mask)

    if node_mask is not None:
        save_dict["node_mask"] = np.array(node_mask)

    np.savez_compressed(str(path_prefix) + str(node) + ".npz", **save_dict)


def load_gnn_explanation(path_prefix, node):
    save = np.load(str(path_prefix) + str(node) + ".npz")

    saved_node = save["node"]
    if saved_node != node:
        raise ValueError("Other node then specified", saved_node, node)

    feature_mask = torch.Tensor(save["feature_mask"])
    if "edge_mask" in save:
        edge_mask = torch.Tensor(save["edge_mask"])
    else:
        edge_mask = None

    if "node_mask" in save:
        node_mask = torch.Tensor(save["node_mask"])
    else:
        node_mask = None

    return feature_mask, edge_mask, node_mask


def five_random_closest_neighbors(explainer_instance, node, data_full):
    from torch_geometric.utils import k_hop_subgraph
    import networkx as nx
    import random

    # get neighbors with subset - and without relabeling
    subset, edge_index, _, _ = k_hop_subgraph(
        node, explainer_instance.__num_hops__(), data_full.edge_index, relabel_nodes=False,
        num_nodes=None, flow=explainer_instance.__flow__())
    num_features = data_full.x.size(1)
    num_nodes = subset.size()

    f_mask = torch.ones(num_features)

    n_mask = torch.zeros(num_nodes)

    node_explanation = {node}
    subset = list(subset.numpy())

    graph = nx.Graph()
    graph.add_nodes_from(subset)
    graph.add_edges_from(edge_index.numpy().transpose())

    neighbors = set(graph.neighbors(node))
    for _ in range(6):
        if len(neighbors) > 5 - len(node_explanation):
            break
        node_explanation = node_explanation.union(neighbors)

        new_neighbors = set()
        for neighbor in neighbors:
            new_neighbors = new_neighbors.union(graph.neighbors(neighbor))
        neighbors = new_neighbors.difference(node_explanation)
    node_explanation = node_explanation.union(random.sample(neighbors, min(len(neighbors), 5 - len(node_explanation))))

    if len(node_explanation) < 5:
        print("Found only " + str(len(node_explanation)) + " neighbors for node " + str(node))

    for selected_neighbor in node_explanation:
        n_mask[subset.index(selected_neighbor)] = 1

    return f_mask, n_mask


if __name__ == "__main__":

    # datasets = [
    #     # "Cora",
    #     # "CiteSeer",
    #     # "PubMed",
    #     # "AmazonC"
    #     # "syn2",
    #
    # ]
    # models = [
    #     # "GCN",
    #     # "GAT",
    #     # "GINConv",
    #     # # "APPNP",
    #     # "APPNP2Net"
    # ]
    # config_name, datasets, models, epochs, explainers = "syn2_faith", \
    #                                                     ["syn2_4", ], \
    #                                                     ["GCN_syn2", ], \
    #                                                     [-1, 0, 200, 400, 600, 1400], \
    #                                                     ["five_neighbors", "GNNExplainer", "Grad", "GradInput"]

    # config_name, datasets, models, epochs, explainers = "real", \
    #                                                     ["Cora", "CiteSeer", "PubMed", ], \
    #                                                     ["GCN", "GAT", "GINConv", "APPNP2Net"], \
    #                                                     [-1, ], \
    #                                                     ["five_neighbors", "GNNExplainer", "Grad", "GradInput"]

    config_name, datasets, models, epochs, explainers = "retrain", \
                                                        ["Cora",], \
                                                        ["GCN",], \
                                                        [-1, ], \
                                                        ["GNNExplainer", "Grad", "GradInput"]

    raw_time = {
        "explainer": [],
        "dataset": [],
        "model": [],
        "epoch": [],
        "node": [],
        "time": [],
    }

    working_directory = Path(".").resolve()
    global_results_directory = working_directory.joinpath("results_baselines")
    global_model_save_directory = working_directory.joinpath("results")
    global_results_directory.mkdir(parents=False, exist_ok=True)

    create_mode = True

    evaluate_retrieved_explanations = False
    add_trivial_explanations = False

    for dataset_name in datasets:
        for model_name in models:
            for epoch in epochs:
                for explainer in explainers:
                    explanations = {}
                    dataset, data, results_path = load_dataset(dataset_name, working_directory=working_directory)

                    results_path += "_" + model_name
                    results_directory = global_results_directory.joinpath(explainer).joinpath(results_path)
                    results_directory.mkdir(parents=True, exist_ok=True)

                    model_save_directory = global_model_save_directory.joinpath(results_path)

                    model_classes = {"GCN": GCNNet,
                                     "GAT": GATNet,
                                     "GINConv": GINConvNet,
                                     "APPNP2Net": APPNP2Net,
                                     "GCN_syn2": GCN_syn2,
                                     }
                    model_class = model_classes[model_name]

                    if epoch == -1:
                        path_to_saved_model = str(model_save_directory.joinpath(MODEL_SAVE_NAMES[model_name] + ".pt"))
                    else:
                        path_to_saved_model = str(
                            model_save_directory.joinpath(
                                MODEL_SAVE_NAMES[model_name] + "_epoch_" + str(epoch) + ".pt"))

                    path_to_saved_explanation_prefix = MODEL_SAVE_NAMES[model_name]

                    if epoch != -1:
                        path_to_saved_explanation_prefix += "_epoch_" + str(epoch)

                    path_to_saved_explanation_prefix += "_" + explainer.lower() + "_soft_masks_node_"

                    path_to_saved_explanation_prefix = results_directory.joinpath(path_to_saved_explanation_prefix)

                    model = model_class(dataset)
                    model.to(device)
                    data = data.to(device)

                    load_model(path_to_saved_model, model)
                    print("Loaded saved model")

                    gnn_explainer = GNNExplainer(model, log=False)

                    if dataset_name[:4] == "syn2":
                        selected_nodes = list(range(300, 700)) + list(range(1000, 1400))
                    else:
                        selected_nodes = np.load(global_model_save_directory.joinpath(dataset_name + "_selected_nodes.npy"))

                    if config_name == "retrain":
                        selected_nodes = np.array(range(data.num_nodes))[data.train_mask]

                    # cast to same data format as before and limit to specified block
                    selected_nodes = set(int(node) for node in selected_nodes)
                    # selected_nodes = range(2708)  # setting corra full
                    print("Selected nodes: " + str(selected_nodes))

                    for i, node in enumerate(selected_nodes):
                        start_time = time.time()

                        node_mask = None
                        edge_mask = None

                        if explainer == "GNNExplainer":
                            try:
                                feature_mask, edge_mask, node_mask = load_gnn_explanation(
                                    path_to_saved_explanation_prefix,
                                    node)
                                create_mode = False
                            except FileNotFoundError:
                                feature_mask, edge_mask = gnn_explainer.explain_node(node, data.x, data.edge_index)
                                save_gnn_explanation(path_to_saved_explanation_prefix, node, feature_mask, edge_mask)
                        elif explainer == "GradEdge":
                            computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
                                gnn_explainer.__subgraph__(node, data.x, data.edge_index)
                            try:
                                feature_mask, edge_mask, node_mask = load_gnn_explanation(
                                    path_to_saved_explanation_prefix,
                                    node)
                                create_mode = False
                            except FileNotFoundError:
                                feature_mask, edge_mask = grad_edge_explanation(model,
                                                                                mapping,
                                                                                computation_graph_feature_matrix,
                                                                                computation_graph_edge_index)
                                save_gnn_explanation(path_to_saved_explanation_prefix, node, feature_mask, edge_mask)

                                edge_mask = torch.tensor(edge_mask)
                                feature_mask = torch.tensor(feature_mask)

                            edge_mask_long = torch.zeros(data.edge_index.shape[1])
                            edge_mask_long[hard_edge_mask] = edge_mask
                            edge_mask = edge_mask_long
                        elif explainer in ["Grad", "GradInput"]:
                            try:
                                feature_mask, edge_mask, node_mask = load_gnn_explanation(
                                    path_to_saved_explanation_prefix,
                                    node)
                                create_mode = False
                            except FileNotFoundError:
                                computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
                                    gnn_explainer.__subgraph__(node, data.x, data.edge_index)
                                if explainer == "Grad":
                                    feature_mask, node_mask = grad_node_explanation(model,
                                                                                    mapping,
                                                                                    computation_graph_feature_matrix,
                                                                                    computation_graph_edge_index)
                                elif explainer == "GradInput":
                                    feature_mask, node_mask = gradinput_node_explanation(model,
                                                                                         mapping,
                                                                                         computation_graph_feature_matrix,
                                                                                         computation_graph_edge_index)
                                else:
                                    raise NotImplementedError("")
                                save_gnn_explanation(path_to_saved_explanation_prefix, node, feature_mask,
                                                     node_mask=node_mask)
                                feature_mask = torch.tensor(feature_mask)
                                node_mask = torch.tensor(node_mask)

                        elif explainer == "five_neighbors":
                            try:
                                feature_mask, edge_mask, node_mask = load_gnn_explanation(
                                    path_to_saved_explanation_prefix,
                                    node)
                                create_mode = False
                            except FileNotFoundError:
                                feature_mask, node_mask = five_random_closest_neighbors(gnn_explainer, node, data)
                                save_gnn_explanation(path_to_saved_explanation_prefix, node, feature_mask,
                                                     node_mask=node_mask)
                        else:
                            raise NotImplementedError("Explainer not implemented")

                        end_time = time.time()

                        raw_time["explainer"].append(explainer)
                        raw_time["dataset"].append(dataset_name)
                        raw_time["model"].append(model_name)
                        raw_time["epoch"].append(epoch)
                        raw_time["node"].append(node)
                        raw_time["time"].append(end_time - start_time)

                        if i % 10 == 0:
                            print("\n", explainer, dataset_name, model_name, epoch, i, end=" ")

                        explanations[node] = feature_mask, edge_mask, node_mask
                    print("")

                    if evaluate_retrieved_explanations:
                        print("Starting evaluation", time.ctime())
                        evaluate_explanations(explainer, model_name, dataset_name, model, data, explanations,
                                              global_results_directory, epoch)

                        if dataset_name[:4] == "syn2":
                            evaluate_synthetic(explainer, model_name, dataset_name, model, data, explanations,
                                              global_results_directory, epoch)
                        print("Finished evaluation", time.ctime())

                if add_trivial_explanations:
                    print("Adding trivial explanations")
                    empty_explanations = {}
                    for node in explanations:
                        feature_mask, _, node_mask = explanations[node]

                        feature_mask = torch.zeros_like(feature_mask)
                        edge_mask = None
                        node_mask = torch.zeros_like(node_mask)

                        empty_explanations[node] = feature_mask, edge_mask, node_mask

                    print("Starting evaluation", time.ctime())
                    evaluate_explanations("edge_only", model_name, dataset_name, model, data, empty_explanations,
                                          global_results_directory, epoch)
                    if dataset_name[:4] == "syn2":
                        evaluate_synthetic("edge_only", model_name, dataset_name, model, data, empty_explanations,
                                           global_results_directory, epoch)
                    print("Finished evaluation", time.ctime())
                print("")

    if create_mode:
        df_time = pd.DataFrame(data=raw_time)
        df_time.to_csv(global_results_directory.joinpath(config_name + "_time.csv"))

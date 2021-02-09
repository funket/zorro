from pathlib import Path
import numpy as np
import time
import torch
import pandas as pd

from print_progress_overview import get_combinations, DATASET_RESULT_PATHS, MODEL_SAVE_NAMES
from explainer import load_minimal_nodes_and_features_sets
from models import load_model, load_dataset, GCNNet, GATNet, APPNP2Net, GINConvNet, GCN_syn2
from evaluation import evaluate_explanations, evaluate_synthetic

if __name__ == "__main__":
    working_directory = Path(".")
    global_results_directory = working_directory.joinpath("results_evaluated")
    global_model_save_directory = working_directory.joinpath("results")
    combinations = get_combinations()
    device = "cpu"

    for dataset_name, model_name, search_paths in combinations:
        for i, path_pattern in enumerate(search_paths):
            explainer = "Zorro"
            if path_pattern.find("_t_3") != -1:
                explainer += "_t_3"

            epoch = -1
            if path_pattern.find("_epoch_") != -1:
                # retrieve epoch from path
                epoch_pos_start = path_pattern.find("_epoch_") + 7
                epoch_pos_end = path_pattern[epoch_pos_start:].find("_")
                epoch = int(path_pattern[epoch_pos_start:epoch_pos_start + epoch_pos_end])

            print(time.ctime(), explainer, dataset_name, model_name, epoch)

            results_directory = working_directory.joinpath("results")
            results_path_prefix = DATASET_RESULT_PATHS[dataset_name]

            if dataset_name[:4] == "syn2":
                selected_nodes = list(range(300, 700)) + list(range(1000, 1400))
            else:
                selected_nodes = np.load(results_directory.joinpath(dataset_name + "_selected_nodes.npy"))

            results_path = results_path_prefix + "_" + model_name
            results_directory = working_directory.joinpath("results")
            results_directory = results_directory.joinpath(results_path)

            file_prefix = results_directory.joinpath(MODEL_SAVE_NAMES[model_name] + "_explanation")

            minimal_sets = {}
            first_explanations = {}
            skip = False
            for node in selected_nodes:
                node = int(node)
                try:
                    # remove _node_{:d}.npz from path pattern
                    minimal_sets[node] = load_minimal_nodes_and_features_sets(str(file_prefix) + path_pattern[:-14],
                                                                              node)
                except FileNotFoundError:
                    skip = True
                    break

                selected_nodes, selected_features, executed_selections = minimal_sets[node][0]
                selected_nodes = torch.Tensor(selected_nodes.squeeze())
                selected_features = torch.Tensor(selected_features.squeeze())
                first_explanations[node] = selected_features, None, selected_nodes

            if skip:
                continue

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

            model = model_class(dataset)
            model.to(device)
            data = data.to(device)

            load_model(path_to_saved_model, model)

            print("Evaluate")

            if dataset_name[:4] != "syn2":
                evaluate_explanations(explainer, model_name, dataset_name, model, data, first_explanations,
                                      global_results_directory, epoch)
            else:
                evaluate_synthetic(explainer, model_name, dataset_name, model, data, first_explanations,
                                   global_results_directory, epoch)

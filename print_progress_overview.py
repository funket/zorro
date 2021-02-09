from pathlib import Path
from execution import MODEL_SAVE_NAMES
import numpy as np
import time
from itertools import product

DATASET_RESULT_PATHS = {"Cora": "cora",
                        "CiteSeer": "citeseer",
                        "PubMed": "pubmed",
                        "syn2": "syn2",
                        "syn2_1": "syn2_1",
                        "syn2_2": "syn2_2",
                        "syn2_3": "syn2_3",
                        "syn2_4": "syn2_4",
                        }


def get_combinations():
    datasets = ["Cora", "CiteSeer", "PubMed", ]
    models = ["GCN", "GAT", "GINConv", "APPNP2Net", ]

    # check which files are already present
    combinations = []
    raw_combinations = list(product(datasets, models))
    for dataset_name, model in raw_combinations:
        paths = ["_node_{:d}.npz", ]
        path_pattern = "_t_3_r_1_node_{:d}.npz"
        if model == "GCN_syn2":
            path_pattern = "_r_1_node_{:d}.npz"
        paths.append(path_pattern)
        combinations.append((dataset_name, model, paths))

    model = "GCN_syn2"
    paths = ["_node_{:d}.npz", "_t_3_node_{:d}.npz"]
    combinations.append(("syn2", model, paths))
    combinations.append(("syn2_1", model, paths))
    combinations.append(("syn2_2", model, paths))
    combinations.append(("syn2_3", model, paths))
    paths = ["_node_{:d}.npz",
             "_t_3_node_{:d}.npz",
             "_t_3_epoch_0_node_{:d}.npz",
             "_t_3_epoch_200_node_{:d}.npz",
             "_t_3_epoch_400_node_{:d}.npz",
             "_t_3_epoch_600_node_{:d}.npz",
             "_t_3_epoch_1400_node_{:d}.npz",
             ]
    combinations.append(("syn2_4", model, paths))

    return combinations


if __name__ == "__main__":
    total_counter = 0

    combinations = get_combinations()
    for dataset_name, model, search_paths in combinations:
        working_directory = Path(".")

        results_directory = working_directory.joinpath("results")
        results_path_prefix = DATASET_RESULT_PATHS[dataset_name]

        if dataset_name[:4] == "syn2":
            selected_nodes = list(range(300, 700)) + list(range(1000, 1400))
        else:
            selected_nodes = np.load(results_directory.joinpath(dataset_name + "_selected_nodes.npy"))
        results_path = results_path_prefix + "_" + model
        results_directory = working_directory.joinpath("results")
        results_directory = results_directory.joinpath(results_path)

        file_prefix = results_directory.joinpath(MODEL_SAVE_NAMES[model] + "_explanation")

        counter = [0] * len(search_paths)

        for node in selected_nodes:
            for i, path_pattern in enumerate(search_paths):
                try:
                    with open(Path((str(file_prefix) + path_pattern).format(int(node)))):
                        pass
                    counter[i] += 1
                except FileNotFoundError:
                    pass

        print(dataset_name, model, *counter, sep="\t")
        total_counter += sum(counter)

    print(time.ctime(), "Total number of explanations:", total_counter)

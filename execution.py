import logging
import sys
import os
import argparse
import random


def set_up_logger(log_file_path, logger_name="explainer"):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # create file handler which logs even debug messages
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # create handler
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # add stdout
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    return logger


MODEL_SAVE_NAMES = {"GCN": "gcn_2_layers",
                    "GAT": "gat_2_layers",
                    "GINConv": "gin_2_layers",
                    "APPNP2Net": "appnp_2_layers",
                    "GCN_syn2": "gcn_3_layers",
                    }


def get_save_file_path(prefix, node, tau, recursion_depth, full_search, samples, epoch):
    path = str(prefix)
    if tau != 15:
        path += "_t_" + str(tau)
    if recursion_depth != np.inf:
        path += "_r_" + str(recursion_depth)
    if full_search:
        path += "_ng_"
    if samples != 100:
        path += "_n_" + str(samples)
    if epoch != -1:
        path += "_epoch_" + str(epoch)
    path += "_node_" + str(node) + ".npz"
    return Path(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Execute explainer")
    parser.add_argument("-c", "--cpu_only", action="store_true",
                        help="Use CPU calculation even if CUDA and GPU is available")
    parser.add_argument("-g", "--gpu", default=-1, type=int, help="GPU number to execute")
    parser.add_argument("--dataset",
                        default="Cora",  # "Cora",
                        type=str,
                        help="Specify the dataset",
                        choices={
                            "Cora",
                            "CiteSeer",
                            "PubMed",
                            "syn2",
                            "syn2_1",
                            "syn2_2",
                            "syn2_3",
                            "syn2_4",
                        })
    parser.add_argument("--model",
                        default="GCN",
                        type=str,
                        help="Specify the model",
                        choices={"GCN", "GAT", "GINConv", "APPNP2Net", "GCN_syn2"})
    parser.add_argument("--epoch", default=-1, type=int, help="Specify the epoch of the model, -1 means last epoch")
    parser.add_argument("--nnodes", default=10, type=int, help="Specify the number of nodes to explain")
    parser.add_argument("--predefined_nodes", action="store_true",
                        help="Load <wd>/results/<dataset>_selected_nodes.npy as process those nodes")
    parser.add_argument("--offset", default=0, type=int, help="Specify which block of the predefined nodes")
    parser.add_argument("-wd", "--working_directory", type=str, help="Specify path to working directory")
    parser.add_argument("--tau", default=15, type=int, help="Specify tau (threshold)")
    parser.add_argument("--recursion_depth", default=-1, type=int, help="Specify maximum recursion depth")
    parser.add_argument("--full_search", action="store_true", default=False,
                        help="Always check all nodes and all features (non greedy variant)")
    parser.add_argument("--save_initial_improve", action="store_true", default=True,
                        help="Store distortion improve values of first round")
    parser.add_argument("--record_processing_time", action="store_true", default=True,
                        help="Save in addition to selected nodes and features the processing time")
    parser.add_argument("--samples", default=100, type=int, help="Specify samples for fidelity")
    args = parser.parse_args()

    if args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # imports not at start to limit the GPU
    from pathlib import Path
    from explainer import *
    from models import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cpu_only:
        device = torch.device('cpu')

        # limit number of CPU cores
        torch.set_num_threads(16)

    if args.working_directory is None:
        working_directory = Path(".").resolve()
    else:
        working_directory = Path(args.working_directory).resolve()

    global_results_directory = working_directory.joinpath("results")
    global_results_directory.mkdir(parents=False, exist_ok=True)
    dataset, data, results_path = load_dataset(args.dataset, working_directory=working_directory)

    results_path += "_" + args.model
    results_directory = global_results_directory.joinpath(results_path)
    results_directory.mkdir(parents=False, exist_ok=True)

    model_classes = {"GCN": GCNNet,
                     "GAT": GATNet,
                     "GINConv": GINConvNet,
                     "APPNP10Net": APPNP10Net,
                     "APPNP2Net": APPNP2Net,
                     "GCN_syn2": GCN_syn2,
                     }
    model_class = model_classes[args.model]

    if args.epoch == -1:
        path_to_saved_model = str(results_directory.joinpath(MODEL_SAVE_NAMES[args.model] + ".pt"))
    else:
        path_to_saved_model = str(
            results_directory.joinpath(MODEL_SAVE_NAMES[args.model] + "_epoch_" + str(args.epoch) + ".pt"))

    path_to_saved_explanation_prefix = results_directory.joinpath(MODEL_SAVE_NAMES[args.model] + "_explanation")

    past_execution_counter = 0
    while True:
        path_to_log_file = results_directory.joinpath(args.model + "_execution_" + str(past_execution_counter) + ".log")
        try:
            with open(path_to_log_file):
                pass
        except FileNotFoundError:
            break

        past_execution_counter += 1

    if args.recursion_depth == -1:
        recursion_depth = np.inf
    else:
        recursion_depth = args.recursion_depth

    tau = args.tau / 100

    logger = set_up_logger(path_to_log_file)
    logger.info("Working directory: " + str(working_directory))
    logger.info("CPU only: " + str(args.cpu_only))
    logger.info("GPU Number: " + str(args.model))
    logger.info("Device: " + str(device))
    logger.info("Model: " + args.model)
    if args.epoch != -1:
        logger.info("Epoch: " + str(args.epoch))
    logger.info("Dataset: " + args.dataset)
    logger.info("Tau: " + str(tau))
    logger.info("Recursion depth: " + str(recursion_depth))
    logger.info("Samples: " + str(args.samples))
    if args.predefined_nodes:
        logger.info("Load predefined nodes: True")
    else:
        logger.info("Load predefined nodes: False")
        logger.info("Number of Nodes: " + str(args.nnodes))

    model = model_class(dataset)
    model.to(device)
    data = data.to(device)

    try:
        load_model(path_to_saved_model, model)
        logger.info("Loaded saved model")
    except FileNotFoundError:
        if args.epoch != -1:
            raise Exception("Not supported if epoch is not last")

        if args.dataset[:4] == "syn2" or args.dataset == "syn1":
            if args.dataset == "syn2_4":
                accuracies = train_model(model, data, epochs=2000, lr=0.001, weight_decay=0.005, clip=2.0,
                                         loss_function="cross_entropy", epoch_save_path=path_to_saved_model)
                np.savez_compressed(str(results_directory.joinpath("accuracies.npz")),
                                    **{"accuracies": np.array(accuracies)})
            else:
                train_model(model, data, epochs=2000, lr=0.001, weight_decay=0.005, clip=2.0,
                            loss_function="cross_entropy")
        else:
            train_model(model, data)
        logger.info("Finished training model")
        save_model(model, path_to_saved_model)
        logger.info("Saved model")

    logger.info(retrieve_accuracy(model, data))

    explainer = Zorro(model, device, greedy=not args.full_search,
                      record_process_time=args.record_processing_time, samples=args.samples)

    if args.dataset == "syn1":
        explainer.add_noise = True

    total_number_of_nodes, _ = data.x.size()
    if args.predefined_nodes:
        if args.dataset == "syn1":
            selected_nodes = list(range(300, 700))
        elif args.dataset[:4] == "syn2":
            selected_nodes = list(range(300, 700)) + list(range(1000, 1400))
        else:
            selected_nodes = np.load(global_results_directory.joinpath(args.dataset + "_selected_nodes.npy"))

        if args.dataset == "Cora":
            # select training nodes
            selected_nodes = np.array(range(data.num_nodes))[data.train_mask.cpu().numpy()]

        # cast to same data format as before and limit to specified block
        selected_nodes = set(int(node) for node in selected_nodes[args.offset:args.offset + args.nnodes])
        logger.info("Selected nodes: " + str(selected_nodes))
    elif args.nnodes < total_number_of_nodes:
        selected_nodes = set()
        possible_nodes = list(range(total_number_of_nodes))
        while len(selected_nodes) < args.nnodes and len(possible_nodes) > 0:
            possible_node = random.choice(possible_nodes)
            # check if explanation does not exists
            try:
                with open(
                        get_save_file_path(path_to_saved_explanation_prefix, possible_node, args.tau, recursion_depth,
                                           args.full_search, args.samples, args.epoch)):
                    pass
            except FileNotFoundError:
                possible_nodes.remove(possible_node)
                selected_nodes.add(possible_node)

        logger.info("Selected nodes: " + str(selected_nodes))
    else:
        selected_nodes = range(total_number_of_nodes)
        logger.info("Selected nodes: all")

    for node in selected_nodes:
        explanation_save_path = get_save_file_path(path_to_saved_explanation_prefix, node, args.tau, recursion_depth,
                                                   args.full_search, args.samples, args.epoch)
        # skip existing explanations
        try:
            with open(explanation_save_path):
                continue
        except FileNotFoundError:
            pass

        explanation = explainer.explain_node(node, data.x, data.edge_index,
                                             tau=tau,
                                             recursion_depth=recursion_depth,
                                             save_initial_improve=args.save_initial_improve)
        if args.save_initial_improve:
            save_minimal_nodes_and_features_sets(explanation_save_path, node, explanation[0], explanation[1],
                                                 explanation[2])
        else:
            save_minimal_nodes_and_features_sets(explanation_save_path, node, explanation)

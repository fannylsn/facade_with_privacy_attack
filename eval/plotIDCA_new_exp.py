import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append("/eval/")
from plotIDCA import (
    compute_rates,
    plot_cluster_model_evolution,
    plot_cluster_variation,
    plot_final_cluster_model_attribution,
)

plt.rcParams.update({"font.size": 20})
plt.rc("legend", fontsize=18)
# matplotlib.rc("xtick", labelsize=20)
# matplotlib.rc("ytick", labelsize=20)


def assign_cte(path):
    global NODE
    global CONFIGS
    global NUM_NODES
    global EVAL
    global NODE_TYPES
    global ACCRONYM
    global COLORS
    global LINE_STYLES

    if "imagenette" in path or "imgnette" in path:
        NODE = "IMAGENETTE"
    elif "cifar" in path:
        NODE = "CIFAR"
    elif "flickr" in path:
        NODE = "FLICKR-MAMMALS"

    if NODE == "IMAGENETTE":
        CONFIGS = {
            "config1": "12:12",
            "config2": "16:8",
            "config3": "20:4",
        }
        NUM_NODES = 24
        EVAL = "test"
    elif NODE == "CIFAR":
        CONFIGS = {
            "config1": "16:16",
            "config2": "24:8",
            "config3": "30:2",
        }
        NUM_NODES = 32
        EVAL = "test"
    elif NODE == "FLICKR-MAMMALS":
        CONFIGS = {
            "config1": "8:8",
            "config3": "14:2",
        }
        NUM_NODES = 16
        EVAL = "val"
    NODE_TYPES = ["DEPRL", "DAC", "EL", "FACADE (ours)"]  # , "DPSGD_5" "IFCA_RESTARTS", "IFCA"
    ACCRONYM = {
        "idca": "FACADE (ours)",
        # "ifca_restarts": "IFCA_RESTARTS",
        # "ifca": "IFCA",
        # "dpsgd_5": "DPSGD_5",
        "dpsgd": "EL",
        "deprl": "DEPRL",
        "dac": "DAC",
    }
    COLORS = {
        "FACADE (ours)": "b",
        # "IFCA": "orange",
        # "IFCA_RESTARTS": "darkorange",
        "EL": "g",
        # "EL_5": "lime",
        "DEPRL": "r",
        "DAC": "purple",
    }
    # colorcblind friendly
    COLORS = {
        "FACADE (ours)": "#E34234",  # Vermillion
        "EL": "#009B95",  # Bluish Green
        "DEPRL": "#00BFFF",  # Sky Blue
        "DAC": "#E69F00",  # Orange
    }

    LINE_STYLES = ["-", "-"]


def plot_results(parent_folder_path, data_machine="machine0", data_node=0):
    assign_cte(parent_folder_path)

    subdirs = os.listdir(parent_folder_path)
    subdirs.sort()

    # extract all experiment and make a big dict with all the results
    all_exp_results = {}
    multi_model = {}
    exp_subdirs = {}
    for subdir in subdirs:
        folder_path = os.path.join(parent_folder_path, subdir)
        if not os.path.isdir(folder_path):
            continue
        node_type = None
        for accro, name in ACCRONYM.items():
            if accro.lower() in str(subdir).lower():
                node_type = name
                break
        if node_type is None:
            continue
        print("Reading the folder: ", folder_path)
        print("Node type: ", node_type)
        all_results, has_multi_model = get_data_from_exp(folder_path)
        all_exp_results[node_type] = all_results
        multi_model[node_type] = has_multi_model
        exp_subdirs[node_type] = subdir

    exp_subdirs = OrderedDict((key, exp_subdirs[key]) for key in NODE_TYPES if key in exp_subdirs)
    all_exp_results = OrderedDict((key, all_exp_results[key]) for key in NODE_TYPES if key in all_exp_results)

    # clean in case of missing data
    common_configs = set(all_exp_results[next(iter(all_exp_results))].keys())
    for exp in all_exp_results.values():
        common_configs &= set(exp.keys())
    all_exp_results = {k: {kk: vv for kk, vv in v.items() if kk in common_configs} for k, v in all_exp_results.items()}
    # compute and plot fair metrics

    # UNCOMMENT !!!
    # fair_metrics = get_fair_metrics(all_exp_results)
    # plot_fair_metric(fair_metrics, parent_folder_path)

    # general plot
    # plot_all_accs(all_exp_results, parent_folder_path)

    # plotting across iterations
    fig_acc, axs_acc = plt.subplots(len(common_configs), 1, figsize=(10, 1 + 4 * len(CONFIGS)))
    fig_acc.suptitle("Accuracy for each experiment")
    fig_acc_clust, axs_acc_clust = plt.subplots(len(common_configs), 1, figsize=(10, 1 + 4 * len(CONFIGS)))
    fig_acc_clust.suptitle("Accuracy detailed by cluster")
    fig_acc_sep, axs_acc_sep = plt.subplots(len(common_configs), 2, figsize=(10, 1 + 4 * len(CONFIGS)))
    fig_acc_sep.suptitle(f"Mean accuracy for each cluster\n{NODE}, {NUM_NODES} nodes\n")

    fig_fair_acc, axs_fair_acc = plt.subplots(len(common_configs), 1, figsize=(10, 1 + 4 * len(CONFIGS)))
    fig_fair_acc.suptitle(f"Fair accuracy for each experiment\n{NODE}, {NUM_NODES} nodes")

    fig_acc_bytes, axs_acc_bytes = plt.subplots(len(common_configs), 1, figsize=(10, 1 + 4 * len(CONFIGS)), sharex=True)
    # fig_acc_bytes, axs_acc_bytes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
    for ax in axs_acc_bytes:
        ax.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=True)
    fig_acc_bytes.suptitle(f"Accuracy vs. data transferred (Bytes)\n{NODE}, {NUM_NODES} nodes")

    rows_list = []
    for exp, all_results in all_exp_results.items():
        acc_per_config, acc_per_config_per_cluster = extract_acc_stats(all_results)
        fair_acc_per_config = get_fair_accuracy(acc_per_config_per_cluster)
        acc_per_bytes_per_config = get_acc_by_bytes(all_results)

        # pandas df
        for (config, data), data_all, data_fair in zip(
            acc_per_config_per_cluster.items(), acc_per_config.values(), fair_acc_per_config.values()
        ):
            for cluster, (iters, values, std) in data.items():
                rows_list.append(
                    {
                        "exp": exp,
                        "config": config,
                        "cluster": cluster,
                        "iter": iters[-1],
                        "mean": values[-1],
                        "std": std[-1],
                    }
                )
            iters, values, std = data_all
            rows_list.append(
                {
                    "exp": exp,
                    "config": config,
                    "cluster": "all",
                    "iter": iters[-1],
                    "mean": values[-1],
                    "std": std[-1],
                }
            )
            iters, values, std = data_fair
            rows_list.append(
                {
                    "exp": exp,
                    "config": config,
                    "cluster": "fair_acc",
                    "iter": iters[-1],
                    "mean": values[-1],
                    "std": std[-1],
                }
            )

        # figures
        plt.close()
        plt.figure(fig_acc)
        plot_acc_per_exp(parent_folder_path, acc_per_config, exp, axs_acc)
        plt.close()
        plt.figure(fig_acc_clust)
        plot_acc_per_cluster_per_exp(parent_folder_path, acc_per_config_per_cluster, exp, axs_acc_clust)
        plt.close()
        plt.figure(fig_acc_sep)
        plot_acc_per_per_exp_separated(parent_folder_path, acc_per_config_per_cluster, exp, axs_acc_sep, majority=True)
        plot_acc_per_per_exp_separated(parent_folder_path, acc_per_config_per_cluster, exp, axs_acc_sep, majority=False)
        plt.close()

        plt.figure(fig_fair_acc)
        plot_fair_acc_per_exp(parent_folder_path, fair_acc_per_config, exp, axs_fair_acc)
        plt.close()

        plt.figure(fig_acc_bytes)
        plot_acc_per_bytes(parent_folder_path, acc_per_bytes_per_config, exp, axs_acc_bytes)
        plt.close()

        for i, (config_folder, data) in enumerate(all_results.items()):
            for j, (seed_folder, seed_data) in enumerate(data.items()):
                exp_path = os.path.join(parent_folder_path, exp_subdirs[exp], config_folder, seed_folder)
                if multi_model[exp]:
                    plot_cluster_model_evolution(exp_path, seed_data)
                    plot_cluster_variation(exp_path, seed_data)
                    plot_final_cluster_model_attribution(exp_path, seed_data)

    # save df
    df = pd.DataFrame(rows_list)
    df.to_csv(os.path.join(parent_folder_path, "acc_per_cluster_per_config.csv"))

    # modif legend
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig_acc_sep.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig_acc_sep.legend(lines, labels, loc="upper center", ncol=4)
    plt.figure(fig_acc_sep)
    if NODE == "FLICKR-MAMMALS":
        handles, labels = axs_acc_sep[0, 0].get_legend_handles_labels()
        fig_acc_sep.legend(handles, labels, loc=(0.09, 0.85), ncols=4)
    else:
        handles, labels = axs_acc_sep[0, 0].get_legend_handles_labels()
        fig_acc_sep.legend(handles, labels, loc=(0.09, 0.88), ncols=4)
    # plt.figlegend(handles, ["1", "2", "3", "4"], loc="lower center", ncols=5, labelspacing=0.0)
    plt.savefig(os.path.join(parent_folder_path, "acc_per_config_sep.pdf"), dpi=300)
    plt.savefig(os.path.join(parent_folder_path, "acc_per_config_sep.png"), dpi=300)


def get_data_from_exp(folder_path):
    config_folders = os.listdir(folder_path)
    config_folders.sort()
    all_results = {}
    for config_folder in config_folders:
        config_folder_path = Path(os.path.join(folder_path, config_folder))
        if not config_folder_path.is_dir():
            continue
        seed_folders = os.listdir(config_folder_path)
        seed_folders.sort()
        config_results = {}
        for seed_folder in seed_folders:
            seed_folder_path = Path(os.path.join(config_folder_path, seed_folder))

            results = []
            machine_folders = os.listdir(seed_folder_path)
            for machine_folder in machine_folders:
                mf_path = os.path.join(seed_folder_path, machine_folder)
                if not os.path.isdir(mf_path):
                    continue

                files = os.listdir(mf_path)
                files = [f for f in files if f.endswith("_results.json")]
                files = [f for f in files if not f.startswith("-1")]  # remove server in IFCA
                for f in files:
                    filepath = os.path.join(mf_path, f)
                    with open(filepath, "r") as inf:
                        results.append(json.load(inf))
                        # print(inf)
                        # print(results[-1]["test_acc"].keys())
                        has_multi_model = "test_best_model_idx" in results[-1]
            config_results[seed_folder] = results
        all_results[config_folder] = config_results
    return all_results, has_multi_model


def get_fair_metrics(all_exp_results):
    all_configs = sorted(list(set([config for exp_data in all_exp_results.values() for config in exp_data.keys()])))
    all_test_metrics = {m: {config: {} for config in all_configs} for m in METRIC_FUNCS.keys()}

    for node_type, node_data in all_exp_results.items():
        for config, data in node_data.items():
            # process json strings
            for results in data.values():
                for res in results:
                    res["per_sample_pred_test"] = {
                        k: (v if isinstance(v, list) else json.loads(v)) for k, v in res["per_sample_pred_test"].items()
                    }
                    res["per_sample_true_test"] = {
                        k: (v if isinstance(v, list) else json.loads(v)) for k, v in res["per_sample_true_test"].items()
                    }
            for metric in METRIC_FUNCS.keys():
                temp_metric = []
                # iterate on all same exp with different seeds
                for results in data.values():
                    # now results is the results of one exp, with config config
                    # print(f"Computing {metric} for {config} {node_type}")
                    per_class_rates, per_cluster_rates = compute_rates(results)
                    if per_class_rates is None:
                        continue
                    temp_metric.append(METRIC_FUNCS[metric](per_cluster_rates, per_class_rates))  # call to correct func
                print(f"Node type: {node_type}, config: {config}, metric: {metric} : {np.mean(temp_metric)}")
                all_test_metrics[metric][config][node_type] = {"mean": np.mean(temp_metric), "std": np.std(temp_metric)}
    return all_test_metrics


def plot_fair_metric(all_test_metrics, out_folder):
    for metric in METRIC_FUNCS.keys():
        configs = list(all_test_metrics[metric].keys())
        node_types = list(all_test_metrics[metric][configs[0]].keys())
        values = [
            [all_test_metrics[metric][config][node_type]["mean"] for config in configs] for node_type in node_types
        ]

        std_dev = [
            [all_test_metrics[metric][config][node_type]["std"] for config in configs] for node_type in node_types
        ]

        # contine
        index = np.arange(len(configs))

        df_val = pd.DataFrame(values, index=node_types, columns=configs)
        df_std = pd.DataFrame(std_dev, index=node_types, columns=configs)
        df = pd.concat([df_val, df_std], keys=["mean", "std"], axis=0)
        df.to_csv(os.path.join(out_folder, f"{metric}_per_node_type_per_config.csv"))

        plt.close()
        plt.figure(figsize=(10, 7))
        for node_type, vals, std in zip(node_types, values, std_dev):
            plt.errorbar(index, vals, std, fmt="o", linewidth=2, capsize=6, label=node_type, color=COLORS[node_type])

        plt.xlabel("Ratio of majority to minority (clusters 0:1)     ")
        plt.ylabel(METRIC_LABEL[metric])
        plt.title(f"{METRIC_LABEL[metric]} by experiment    ")
        plt.xticks(index, map(lambda x: CONFIGS[x], configs))
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{metric}_per_node_type_per_config.pdf"), dpi=300)
        plt.savefig(os.path.join(out_folder, f"{metric}_per_node_type_per_config.png"), dpi=300)


def extract_acc_stats(all_results):
    if EVAL == "test":
        eval_column = "test_acc"
    else:
        eval_column = "validation_acc"
    all_test_accs = {config: {} for config in all_results.keys()}
    acc_per_config = {config: {} for config in all_results.keys()}
    for config, data in all_results.items():
        clusters = set(x["cluster_assigned"] for x in next(iter(data.values())))
        per_seed_acc = {cluster: [] for cluster in clusters}
        per_seed_acc_no_clust = []
        for seed_data in data.values():
            # final_iter = max(int(k) for k in seed_data[0]["test_acc"].keys())
            iters = [int(k) for k in seed_data[0][eval_column].keys()]
            per_clusters_acc = {c: [] for c in clusters}
            for x in seed_data:
                per_clusters_acc[x["cluster_assigned"]].append(x[eval_column].values())

            mean = [np.mean(x) for x in zip(*[x[eval_column].values() for x in seed_data])]
            per_seed_acc_no_clust.append((iters, mean))

            for cluster, v in per_clusters_acc.items():
                mean = [np.mean(x) for x in zip(*v)]
                per_seed_acc[cluster].append((iters, mean))  # mean across all nodes
                # per_seed_acc[cluster].append(sum(v) / len(v))  # mean across all nodes

        # across all clusters
        iters = per_seed_acc_no_clust[0][0]
        values = [np.mean(v) for v in zip(*[x[1] for x in per_seed_acc_no_clust])]
        std = [np.std(v) for v in zip(*[x[1] for x in per_seed_acc_no_clust])]
        if len(values) == len(iters):
            acc_per_config[config] = (per_seed_acc_no_clust[0][0], values, std)

        # detail per cluster
        for cluster, v in per_seed_acc.items():
            iters = v[0][0]
            values = [np.mean(acc) for acc in zip(*[x[1] for x in v])]
            if len(values) != len(iters):
                # meaning: exp was manually concaelled, ignore
                continue
            std = [np.std(acc) for acc in zip(*[x[1] for x in v])]
            all_test_accs[config][cluster] = (iters, values, std)

    return acc_per_config, all_test_accs


def get_fair_accuracy(acc_per_config_per_cluster):
    # fair_acc_per_config = {config: {} for config in acc_per_config_per_cluster.keys()}
    fair_acc_per_config = {}
    for config, data in acc_per_config_per_cluster.items():
        if len(data) != 2:
            raise ValueError("Only two clusters are supported")
        (iters, values_0, std_0) = data[0]
        (_, values_1, std_1) = data[1]
        mean_acc = [(np.mean(x) + np.mean(y)) / 2 for x, y in zip(values_0, values_1)]
        acc_sim = [100 - np.abs(x - y) for x, y in zip(values_0, values_1)]
        # fair_acc = [2 * ma * as_ / (ma + as_) for ma, as_ in zip(mean_acc, acc_sim)]
        alpha = 1 / 3
        fair_acc = [alpha * ma + (1 - alpha) * as_ for ma, as_ in zip(mean_acc, acc_sim)]
        fair_acc_per_config[config] = (iters, fair_acc, [0] * len(iters))
    return fair_acc_per_config


def get_acc_by_bytes(all_results):
    acc_per_config = {config: {} for config in all_results.keys()}
    for config, data in all_results.items():
        acc_per_bytes = []
        bytes_per_node = []
        for seed_data in data.values():
            # final_iter = max(int(k) for k in seed_data[0]["test_acc"].keys())
            iters = [k for k in seed_data[0]["test_acc"].keys()]
            bytes = [[x["total_bytes"][it] for it in iters if it in x["total_bytes"]] for x in seed_data]
            acc = [x["test_acc"].values() for x in seed_data]
            mean_acc = [np.mean(x) for x in zip(*acc)]
            mean_bytes = [np.mean(x) for x in zip(*bytes)]
            while len(mean_acc) > len(mean_bytes):
                mean_acc.pop()

            # all seed have same bytes transfered
            bytes_per_node.append(mean_bytes)
            acc_per_bytes.append(mean_acc)
        bytes = [np.mean(x) for x in zip(*bytes_per_node)]
        mean = [np.mean(x) for x in zip(*acc_per_bytes)]
        std = [np.std(x) for x in zip(*acc_per_bytes)]
        acc_per_config[config] = (bytes, mean, std)
    return acc_per_config


def plot_acc_per_exp(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        if data:
            iters, values, std = data
            axs[i].plot(iters, values, label=f"{exp}", color=COLORS[exp])
            axs[i].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
        print(CONFIGS)
        axs[i].set_title(f"Accuracy for {CONFIGS[config]}")
        axs[i].set_ylabel("Accuracy")
        axs[i].yaxis.set_ticks(np.arange(0, 70 + 1, 10.0))
        if data:
            # axs[i].legend(loc="lower right")
            axs[i].legend(loc="lower right", ncol=3, fancybox=True)  # bbox_to_anchor=(1.05, -0.05)
    # set only last
    axs[i].set_xlabel("Communication rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_config.pdf"), dpi=300)
    plt.savefig(os.path.join(folder_path, "acc_per_config.png"), dpi=300)


def plot_fair_acc_per_exp(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        axs[i].grid(True, color="gray", linestyle="-", linewidth=0.5)
        if data:
            iters, values, std = data
            axs[i].plot(iters, values, label=f"{exp}", color=COLORS[exp])
            axs[i].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
        axs[i].set_title(f"Fair accuracy for {CONFIGS[config]}")
        axs[i].set_ylabel("Fair accuracy")
        axs[i].yaxis.set_ticks(np.arange(0, 80 + 1, 20.0))
        if data:
            # axs[i].legend(loc="lower right")
            axs[i].legend(loc="lower right", ncol=2, fancybox=True)  # bbox_to_anchor=(1.05, -0.05)
    # set only last
    axs[i].set_xlabel("Communication rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "fair_acc_per_config.pdf"), dpi=300)
    plt.savefig(os.path.join(folder_path, "fair_acc_per_config.png"), dpi=300)


def plot_acc_per_bytes(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        axs[i].grid(True, color="gray", linestyle="-", linewidth=0.5)
        if data:
            bytes, values, std = data
            bytes = [x * 1e-9 for x in bytes]
            axs[i].plot(bytes, values, label=f"{exp}", color=COLORS[exp])
            axs[i].fill_between(
                bytes, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
        axs[i].set_title(f"Accuracy for {CONFIGS[config]}")
        axs[i].set_ylabel("Accuracy")
        axs[i].yaxis.set_ticks(np.arange(0, 80 + 1, 20.0))
        if data:
            # axs[i].legend(loc="lower right")
            axs[i].legend(loc="lower right", ncol=2, fancybox=True)  # bbox_to_anchor=(1.05, -0.05)
    # set only last
    axs[i].set_xlabel("Total bytes transfered (Gb)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_bytes_per_config_solo.pdf"), dpi=300)
    plt.savefig(os.path.join(folder_path, "acc_per_bytes_per_config_solo.png"), dpi=300)


def plot_acc_per_cluster_per_exp(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        for clust, cluster_data in data.items():
            iters, values, std = cluster_data
            if clust == 0:
                axs[i].plot(iters, values, label=f"{exp}", color=COLORS[exp], linestyle=LINE_STYLES[clust])
            else:
                axs[i].plot(iters, values, color=COLORS[exp], linestyle=LINE_STYLES[clust])
            axs[i].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
            # axs[i].errorbar(iters, values, std, label=f"Cluster {cluster}")
            if exp == NODE_TYPES[-1]:
                # plot line type just once
                axs[i].plot([], [], label=f"Cluster {clust}", color="k", linestyle=LINE_STYLES[clust])
        axs[i].set_title(f"Accuracy for {CONFIGS[config]}")
        axs[i].set_ylabel("Accuracy")
        axs[i].yaxis.set_ticks(np.arange(0, 80 + 1, 20.0))
        if data:
            # axs[i].legend(loc="lower right")
            axs[i].legend(loc="lower right", ncol=3, fancybox=True)
    # set only last
    axs[i].set_xlabel("Communication rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_clust_per_config.pdf"), dpi=300)
    plt.savefig(os.path.join(folder_path, "acc_per_clust_per_config.png"), dpi=300)


def plot_acc_per_per_exp_separated(folder_path, all_test_accs, exp, axs, majority=True):
    if majority:
        cur_cluster = 0
        label = "majority"
        ticks = np.arange(0, 80 + 1, 20.0)
        column = 0
    else:
        cur_cluster = 1
        label = "minority"
        ticks = np.arange(0, 80 + 1, 20.0)
        column = 1

    for i, (config, data) in enumerate(all_test_accs.items()):
        for clust, cluster_data in data.items():
            iters, values, std = cluster_data
            if clust != cur_cluster:
                continue
            axs[i, column].plot(iters, values, label=f"{exp}", color=COLORS[exp], linestyle=LINE_STYLES[clust])
            axs[i, column].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
        axs[i, column].set_title(f"{CONFIGS[config]}, {label}")
        axs[i, column].grid(True, color="gray", linestyle="-", linewidth=0.5)
        if column == 0:
            axs[i, 0].set_ylabel("Accuracy")
        axs[i, column].yaxis.set_ticks(ticks)
        # if data:
        # axs[i].legend(loc="lower right")
        # axs[i, column].legend(loc="lower right", ncol=3, fancybox=True)
    # set only last
    axs[-1, column].set_xlabel("Communication rounds")
    plt.tight_layout()


def plot_settling_time(folder_path, all_results):
    all_settling_times = {config: [] for config in all_results.keys()}
    for config, data in all_results.items():
        for seed_data in data.values():
            data = [list(x["test_best_model_idx"].values()) for x in seed_data]
            idx = [int(x) for x in seed_data[0]["test_best_model_idx"].keys()]
            variations = []
            for x in data:
                varia = []
                for i in range(1, len(x)):
                    varia.append(int(bool(x[i] - x[i - 1])))
                variations.append(varia)
            total_variations = np.sum(np.array(variations), axis=0)
            # take idx + 1 beacause the ith variation ends at the i+1th iteration
            # second + 1 ??
            settling_iter = idx[np.max(np.nonzero(total_variations)) + 1 + 1] if np.any(total_variations) else 0

            all_settling_times[config].append(settling_iter)

    configs = list(all_settling_times.keys())
    settling_times = [np.array(all_settling_times[config]) for config in configs]

    _, _ = plt.subplots()
    # plt.boxplot(settling_times)
    # plt.violinplot(settling_times, showmeans=False, showmedians=True)
    for i, times in enumerate(settling_times):
        noise_x = np.random.normal(0, 0.01, len(times))
        noise_y = np.random.normal(0, 0.05, len(times))
        plt.scatter(i + 1 + noise_x, times + noise_y, alpha=0.5)
    plt.xlabel("Ratio of majority to minority")
    plt.ylabel("Settling iteration")
    plt.title("Settling iteration by experiment")
    plt.xticks(range(1, len(configs) + 1), map(lambda x: CONFIGS[x], configs))

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "settling_time_per_config.pdf"), dpi=300)
    plt.savefig(os.path.join(folder_path, "settling_time_per_config.png"), dpi=300)


def compute_demo_parity(per_clust_rates, per_class_rates):
    tot_demo_par = 0
    clusters = list(per_clust_rates.keys())
    classes = list(range(len(per_class_rates[clusters[0]]["TP"])))
    for class_ in classes:
        pos_preds_0 = per_class_rates[clusters[0]]["TP"][class_] + per_class_rates[clusters[0]]["FP"][class_]
        tot_0 = np.sum([per_class_rates[clusters[0]][k][class_] for k in per_class_rates[clusters[0]].keys()])
        pos_preds_1 = per_class_rates[clusters[1]]["TP"][class_] + per_class_rates[clusters[1]]["FP"][class_]
        tot_1 = np.sum([per_class_rates[clusters[1]][k][class_] for k in per_class_rates[clusters[1]].keys()])

        demo_parity = abs(pos_preds_0 / tot_0 - pos_preds_1 / tot_1)
        tot_demo_par += demo_parity
    return tot_demo_par / len(classes)


def compute_equ_oppo(per_clust_rates, per_class_rates):
    tot_equ_oppo = 0
    clusters = list(per_clust_rates.keys())
    classes = list(range(len(per_class_rates[clusters[0]]["TP"])))
    for class_ in classes:
        rec_0 = per_class_rates[clusters[0]]["TP"][class_] / (
            per_class_rates[clusters[0]]["TP"][class_] + per_class_rates[clusters[0]]["FN"][class_]
        )
        rec_1 = per_class_rates[clusters[1]]["TP"][class_] / (
            per_class_rates[clusters[1]]["TP"][class_] + per_class_rates[clusters[1]]["FN"][class_]
        )

        eq_oppo = abs(rec_0 - rec_1)
        tot_equ_oppo += eq_oppo
    return eq_oppo / len(classes)


def compute_equalized_odds(per_clust_rates, per_class_rates):
    tot_equ_odds = 0
    clusters = list(per_clust_rates.keys())
    classes = list(range(len(per_class_rates[clusters[0]]["TP"])))
    for class_ in classes:
        rec_0 = per_class_rates[clusters[0]]["TP"][class_] / (
            per_class_rates[clusters[0]]["TP"][class_] + per_class_rates[clusters[0]]["FN"][class_]
        )
        rec_1 = per_class_rates[clusters[1]]["TP"][class_] / (
            per_class_rates[clusters[1]]["TP"][class_] + per_class_rates[clusters[1]]["FN"][class_]
        )
        eq_oppo = abs(rec_0 - rec_1)

        fpr_0 = per_class_rates[clusters[0]]["FP"][class_] / (
            per_class_rates[clusters[0]]["TN"][class_] + per_class_rates[clusters[0]]["FP"][class_]
        )
        fpr_1 = per_class_rates[clusters[1]]["FP"][class_] / (
            per_class_rates[clusters[1]]["TN"][class_] + per_class_rates[clusters[1]]["FP"][class_]
        )
        temp = abs(fpr_0 - fpr_1)
        tot_equ_odds += temp + eq_oppo

    return tot_equ_odds / len(classes)


METRIC_FUNCS = {"demo_parity": compute_demo_parity, "equ_oppo": compute_equ_oppo, "eq_odds": compute_equalized_odds}
METRIC_LABEL = {"demo_parity": "Demographic parity", "equ_oppo": "Equality of opportunity", "eq_odds": "Equalized odds"}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # The args are:
        # 1: the folder with the data
        plot_results(sys.argv[1])

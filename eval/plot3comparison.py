import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append("/eval/")
from plotIDCA import (
    CIFAR_CLASSES,
    compute_rates,
)

CONFIGS = ["IFCA", "DPSGD", "IDCA", "DEPRL_V1", "DEPRL_V2", "DAC"]
CONFIGS = ["IFCA_NO_GRAD", "IFCA", "IDCA_DEG_3", "IDCA_DEG_4", "IDCA_FC"]


def get_stats(data):
    assert len(data) > 0
    mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}
    for key in data[0].keys():
        all_nodes = [i[key] for i in data]
        all_nodes = np.array(all_nodes)
        mean = np.mean(all_nodes)
        std = np.std(all_nodes)
        min = np.min(all_nodes)
        max = np.max(all_nodes)
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min
        max_dict[int(key)] = max
    return mean_dict, stdev_dict, min_dict, max_dict


def plot(means, stdevs, mins, maxs, title, label, loc, xlabel="communication rounds"):
    plt.title(title)
    plt.xlabel(xlabel)
    x_axis = np.array(list(means.keys()))
    y_axis = np.array(list(means.values()))
    err = np.array(list(stdevs.values()))
    plt.plot(x_axis, y_axis, label=label)
    plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)
    plt.legend(loc=loc)


def replace_dict_key(d_org: dict, d_other: dict):
    result = {}
    for x, y in d_org.items():
        result[d_other[x]] = y
    return result


def plot_results(folder_path, data_machine="machine0", data_node=0):
    print("Reading the folder: ", folder_path)
    folder_path = Path(os.path.abspath(folder_path))
    bytes_means, bytes_stdevs = {}, {}
    meta_means, meta_stdevs = {}, {}
    data_means, data_stdevs = {}, {}

    subdirs = os.listdir(folder_path)
    subdirs.sort()
    each_results = {}
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)

        if not os.path.isdir(subdir_path):
            continue
        results = []
        machine_folders = os.listdir(subdir_path)
        for machine_folder in machine_folders:
            mf_path = os.path.join(subdir_path, machine_folder)
            if not os.path.isdir(mf_path):
                continue
            files = os.listdir(mf_path)
            files = [f for f in files if f.endswith("_results.json")]
            files = [f for f in files if not f.startswith("-1")]  # remove server in IFCA
            for f in files:
                filepath = os.path.join(mf_path, f)
                with open(filepath, "r") as inf:
                    results.append(json.load(inf))
        each_results[subdir] = results
        print("Files", files)

    for subdir, results in each_results.items():
        subdir_path = os.path.join(folder_path, subdir)
        for name in CONFIGS:
            if name.lower() in str(subdir).lower():
                config = name
                break
        # Plotting bytes over time
        plt.figure(10)
        b_means, stdevs, mins, maxs = get_stats([x["total_bytes"] for x in results])
        plot(b_means, stdevs, mins, maxs, "Total Bytes", config, "lower right")
        df = pd.DataFrame(
            {
                "mean": list(b_means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(b_means),
            },
            list(b_means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(
            os.path.join(subdir_path, "total_bytes.csv"),
            index_label="rounds",
        )
        plt.savefig(os.path.join(folder_path, "total_bytes.png"), dpi=300)

        # Plot Training loss
        plt.figure(20)
        means, stdevs, mins, maxs = get_stats([x["train_loss"] for x in results])
        plot(means, stdevs, mins, maxs, "Training Loss", config, "upper right")

        # handle the last artificial iteration for all reduce
        if list(iter(b_means.keys())) != list(iter(means.keys())):
            b_means[list(iter(means.keys()))[-1]] = np.nan

        correct_bytes = [b_means[x] for x in means]

        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
                "total_bytes": correct_bytes,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes", "total_bytes"],
        )
        plt.savefig(os.path.join(folder_path, "train_loss.png"), dpi=300)

        plt.figure(11)
        means = replace_dict_key(means, b_means)
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Training Loss",
            config,
            "upper right",
            "Total Bytes per node",
        )
        plt.savefig(os.path.join(folder_path, "bytes_train_loss.png"), dpi=300)

        df.to_csv(os.path.join(subdir_path, "train_loss.csv"), index_label="rounds")

        # Plot Testing loss
        plt.figure(21)
        means, stdevs, mins, maxs = get_stats([x["test_loss"] for x in results])
        plot(means, stdevs, mins, maxs, "Testing Loss", config, "upper right")
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
                "total_bytes": correct_bytes,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes", "total_bytes"],
        )
        plt.savefig(os.path.join(folder_path, "test_loss.png"), dpi=300)

        plt.figure(12)
        means = replace_dict_key(means, b_means)
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Testing Loss",
            config,
            "upper right",
            "Total Bytes per node",
        )
        plt.savefig(os.path.join(folder_path, "bytes_test_loss.png"), dpi=300)

        df.to_csv(os.path.join(subdir_path, "test_loss.csv"), index_label="rounds")

        # Plot Testing Accuracy
        plt.figure(22)
        means, stdevs, mins, maxs = get_stats([x["test_acc"] for x in results])
        plot(means, stdevs, mins, maxs, "Testing Accuracy", config, "lower right")
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": [len(results)] * len(means),
                "total_bytes": correct_bytes,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes", "total_bytes"],
        )
        plt.savefig(os.path.join(folder_path, "test_acc.png"), dpi=300)

        plt.figure(13)
        means = replace_dict_key(means, b_means)
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Testing Accuracy",
            config,
            "lower right",
            "Total Bytes per node",
        )
        df.to_csv(os.path.join(subdir_path, "test_acc.csv"), index_label="rounds")
        plt.savefig(os.path.join(folder_path, "bytes_test_acc.png"), dpi=300)

        # Collect total_bytes shared
        bytes_list = []
        for x in results:
            max_key = str(max(list(map(int, x["total_bytes"].keys()))))
            bytes_list.append({max_key: x["total_bytes"][max_key]})
        means, stdevs, mins, maxs = get_stats(bytes_list)
        bytes_means[config] = list(means.values())[0]
        bytes_stdevs[config] = list(stdevs.values())[0]

        meta_list = []
        for x in results:
            if x["total_meta"]:
                max_key = str(max(list(map(int, x["total_meta"].keys()))))
                meta_list.append({max_key: x["total_meta"][max_key]})
            else:
                meta_list.append({max_key: 0})
        means, stdevs, mins, maxs = get_stats(meta_list)
        meta_means[config] = list(means.values())[0]
        meta_stdevs[config] = list(stdevs.values())[0]

        data_list = []
        for x in results:
            max_key = str(max(list(map(int, x["total_data_per_n"].keys()))))
            data_list.append({max_key: x["total_data_per_n"][max_key]})
        means, stdevs, mins, maxs = get_stats(data_list)
        data_means[config] = list(means.values())[0]
        data_stdevs[config] = list(stdevs.values())[0]

        # Plot total_bytes
        plt.figure(14)
        plt.title("Data Shared")
        x_pos = np.arange(len(bytes_means.keys()))
        plt.bar(
            x_pos,
            np.array(list(bytes_means.values())) // (1024 * 1024),
            yerr=np.array(list(bytes_stdevs.values())) // (1024 * 1024),
            align="center",
        )
        plt.ylabel("Total data shared in MBs")
        plt.xlabel("Fraction of Model Shared")
        plt.xticks(x_pos, list(bytes_means.keys()))
        plt.savefig(os.path.join(folder_path, "data_shared.png"), dpi=300)

        # Plot stacked_bytes
        plt.figure(15)
        plt.title("Data Shared per Neighbor")
        x_pos = np.arange(len(bytes_means.keys()))
        plt.bar(
            x_pos,
            np.array(list(data_means.values())) // (1024 * 1024),
            yerr=np.array(list(data_stdevs.values())) // (1024 * 1024),
            align="center",
            label="Parameters",
        )
        plt.bar(
            x_pos,
            np.array(list(meta_means.values())) // (1024 * 1024),
            bottom=np.array(list(data_means.values())) // (1024 * 1024),
            yerr=np.array(list(meta_stdevs.values())) // (1024 * 1024),
            align="center",
            label="Metadata",
        )
        plt.ylabel("Data shared in MBs")
        plt.xlabel("Fraction of Model Shared")
        plt.xticks(x_pos, list(meta_means.keys()))
        plt.savefig(os.path.join(folder_path, "parameters_metadata.png"), dpi=300)

        # Plotting cluster-model attribution
        if "test_best_model_idx" in results[0].keys():
            plt.figure(30)
            plot_final_cluster_model_attribution(subdir_path, results)

            plt.figure(31)
            plot_cluster_model_evolution(subdir_path, results)

            plt.figure(32)
            plot_cluster_variation(subdir_path, results)

        # faireness
        if "per_sample_pred_test" in results[0].keys() and results[0]["per_sample_pred_test"]:
            for res in results:
                res["per_sample_pred_test"] = {k: json.loads(v) for k, v in res["per_sample_pred_test"].items()}
                res["per_sample_true_test"] = {k: json.loads(v) for k, v in res["per_sample_true_test"].items()}
            per_class_rates, per_cluster_rates = compute_rates(results)
            plt.figure(40)
            plot_per_class_demographic_parity(per_class_rates, folder_path, config)
            plt.figure(41)
            plot_per_class_equal_opportunities(per_class_rates, folder_path, config)


def plot_cluster_model_evolution(folder_path, results):
    data = [(x["cluster_assigned"], int(k), v) for x in results for k, v in x["test_best_model_idx"].items()]
    clusters = set([x[0] for x in data])
    models = set([x[2] for x in data])
    max_iter = max([el[1] for el in data])
    space_iter = data[1][1] - data[0][1]
    init_iter = data[0][1]
    fig, axs = plt.subplots(len(clusters), 1)
    fig.suptitle("Number of node in each cluster picking each model")
    for i, cluster in enumerate(clusters):
        cluster_data = [x for x in data if x[0] == cluster]
        for model in models:
            model_data = [x for x in cluster_data if x[2] == model]
            # count = sorted(Counter(model_data), key=lambda x: x[1])
            count = sorted(Counter(model_data).items(), key=lambda pair: pair[0][1])
            for j in range(init_iter, max_iter + 1, space_iter):
                if j not in [elem[0][1] for elem in count]:
                    count.append(((cluster, j, model), 0))
            count = sorted(count, key=lambda pair: pair[0][1])
            axs[i].plot(
                [elem[0][1] for elem in count],
                [elem[1] for elem in count],
                label=f"model {model}",
            )
        axs[i].set_ylim(ymin=0)
        axs[i].set_xticks(np.arange(0, max_iter + 1, 10))
        axs[i].title.set_text(f"Cluster {cluster} Data Distribution")
        axs[i].legend(loc="upper right", fontsize="8")
        axs[i].set_ylabel("Nodes")
    # fig.text(0.00, 0.5, "Number of nodes", va="center", rotation="vertical")
    axs[i].set_xlabel("Communication Rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "cluster_model_evolution.png"), dpi=300)
    plt.close()


def plot_final_cluster_model_attribution(folder_path, results):
    max_iter = max([int(iter) for x in results for iter in x["test_best_model_idx"].keys()])

    data = [(x["cluster_assigned"], x["test_best_model_idx"][str(max_iter)]) for x in results]
    df = pd.DataFrame(data, columns=["Cluster Assigned", "Best Model"])
    heatmap_data = df.groupby(["Cluster Assigned", "Best Model"]).size().reset_index(name="Count")
    heatmap_matrix = heatmap_data.pivot(index="Cluster Assigned", columns="Best Model", values="Count").fillna(0)
    sns.heatmap(heatmap_matrix, annot=True, fmt="g", cmap="YlGnBu")
    plt.title("Heatmap of Data Distribution")
    plt.savefig(os.path.join(folder_path, "cluster_model_distribution.png"), dpi=300)
    plt.close()


def plot_cluster_variation(folder_path, results):
    # HORIBLE !
    # breakpoint()
    data = [list(x["test_best_model_idx"].values()) for x in results]
    idx = [int(x) for x in results[0]["test_best_model_idx"].keys()]
    variations = []
    for x in data:
        varia = []
        for i in range(1, len(x)):
            varia.append(int(bool(x[i] - x[i - 1])))
        variations.append(varia)
    variations = np.array(variations)
    plt.plot(idx[1:], np.sum(variations, axis=0))
    plt.xlim(left=0)
    plt.title("Variation in model Selection across all nodes")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Sum of all variations")
    plt.savefig(os.path.join(folder_path, "cluster_variation.png"), dpi=300)
    plt.close()


def plot_per_class_demographic_parity(per_class_rates: Dict[int, Dict[str, int]], folder_path, config):
    """Compute and plot the demographic parity with S the sensitive attribute beeing the cluster belonging of each node.
        TP + FP / all preds
    Args:
        per_class_rates (Dict): _description_
        folder_path (_type_): _description_
    """
    clusters = list(per_class_rates.keys())
    pos_preds_0 = per_class_rates[clusters[0]]["TP"] + per_class_rates[clusters[0]]["FP"]
    tot_0 = np.sum([per_class_rates[clusters[0]][k] for k in per_class_rates[clusters[0]].keys()], axis=0)
    pos_preds_1 = per_class_rates[clusters[1]]["TP"] + per_class_rates[clusters[1]]["FP"]
    tot_1 = np.sum([per_class_rates[clusters[1]][k] for k in per_class_rates[clusters[1]].keys()], axis=0)

    demo_parity = abs(pos_preds_0 / tot_0 - pos_preds_1 / tot_1)

    plt.plot(demo_parity, "o", label=config)
    plt.title("Per class demographic parity")
    plt.ylabel("Absolute difference in accuracy")
    plt.xlabel("Classes")
    plt.legend(loc="upper right")
    plt.xticks(range(len(demo_parity)), [CIFAR_CLASSES[i] for i in range(len(demo_parity))], rotation=45)
    plt.savefig(os.path.join(folder_path, "demographic_parity.png"), dpi=300)


def plot_per_class_equal_opportunities(per_class_rates: Dict[int, Dict[str, int]], folder_path, config):
    """Compute and plot the equal opportunities with S the sensitive attribute beeing the cluster belonging of each node.
    Requires that each group has the same recall.
    TP / (TP + FN) (per cluster true positive rate == recall)
    Args:
        per_class_rates (Dict): _description_
        folder_path (_type_): _description_
    """
    clusters = list(per_class_rates.keys())
    rec_0 = per_class_rates[clusters[0]]["TP"] / (
        per_class_rates[clusters[0]]["TP"] + per_class_rates[clusters[0]]["FN"]
    )
    rec_1 = per_class_rates[clusters[1]]["TP"] / (
        per_class_rates[clusters[1]]["TP"] + per_class_rates[clusters[1]]["FN"]
    )

    eq_op = abs(rec_0 - rec_1)

    plt.plot(eq_op, "o", label=config)
    plt.title("Per class equal opportunities")
    plt.ylabel("Absolute difference in recall")
    plt.xlabel("Classes")
    plt.legend(loc="upper right")
    plt.xticks(range(len(eq_op)), [CIFAR_CLASSES[i] for i in range(len(eq_op))], rotation=45)
    plt.savefig(os.path.join(folder_path, "equal_opportunities.png"), dpi=300)


def plot_parameters(path):
    plt.figure(100)
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        files = [f for f in files if f.endswith("_shared_params.json")]
        for f in files:
            filepath = os.path.join(folder_path, f)
            print("Working with ", filepath)
            with open(filepath, "r") as inf:
                loaded_dict = json.load(inf)
                del loaded_dict["order"]
                del loaded_dict["shapes"]
            assert len(loaded_dict["0"]) > 0
            assert "0" in loaded_dict.keys()
            counts = np.zeros(len(loaded_dict["0"]))
            for key in loaded_dict.keys():
                indices = np.array(loaded_dict[key])
                counts = np.pad(
                    counts,
                    max(np.max(indices) - counts.shape[0], 0),
                    "constant",
                    constant_values=0,
                )
                counts[indices] += 1
            plt.plot(np.arange(0, counts.shape[0]), counts, ".")
        print("Saving scatterplot")
        plt.savefig(os.path.join(folder_path, "shared_params.png"))


if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: the folder with the data
    plot_results(sys.argv[1])
    # plot_parameters(sys.argv[1])

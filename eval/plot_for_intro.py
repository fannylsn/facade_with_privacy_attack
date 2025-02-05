import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

plt.rcParams.update({"font.size": 20})
plt.rc("legend", fontsize=18)

CIFAR_CLASSES = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


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


def get_per_cluster_stats(results, metric="test_loss"):
    assert len(results) > 0
    clusters = set([int(x["cluster_assigned"]) for x in results])
    final_data = {}
    for clust in clusters:
        mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}
        for key in results[0][metric].keys():
            # key == iter
            all_nodes = [
                i[metric][key] for i in results if i["cluster_assigned"] == clust
            ]
            all_nodes = np.array(all_nodes)
            mean = np.mean(all_nodes)
            std = np.std(all_nodes)
            min = np.min(all_nodes)
            max = np.max(all_nodes)
            mean_dict[int(key)] = mean
            stdev_dict[int(key)] = std
            min_dict[int(key)] = min
            max_dict[int(key)] = max
        final_data[clust] = [mean_dict, stdev_dict, min_dict, max_dict]
    return final_data


def plot(means, stdevs, mins, maxs, title, label, loc, xlabel="communication rounds"):
    plt.title(title)
    plt.xlabel(xlabel)
    x_axis = np.array(list(means.keys()))
    y_axis = np.array(list(means.values()))
    err = np.array(list(stdevs.values()))
    plt.plot(x_axis, y_axis, label=label)
    plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)
    plt.legend(loc=loc)


def per_cluster_plot(final_data, title, loc, xlabel="communication rounds"):
    for clust, data in final_data.items():
        means, stdevs, mins, maxs = data
        x_axis = np.array(list(means.keys()))
        y_axis = np.array(list(means.values()))
        err = np.array(list(stdevs.values()))
        plt.plot(x_axis, y_axis, label=f"cluster {clust}")
        plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(loc=loc)


def per_cluster_plot_intro(final_data, title, loc, xlabel="communication rounds"):
    means_c = []
    for clust, data in final_data.items():
        means, stdevs, mins, maxs = data
        x_axis = np.array(list(means.keys()))
        y_axis = np.array(list(means.values()))
        means_c.append(y_axis)
        # err = np.array(list(stdevs.values()))
    plt.plot(x_axis, means_c[0], label="majority cluster", color="g")
    plt.plot(x_axis, means_c[1], label="minority cluster", color="limegreen")
    plt.fill_between(
        x_axis, means_c[0], means_c[1], alpha=0.3, color="gray", label="Accuracy gap"
    )
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout()


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

    results = []
    machine_folders = os.listdir(folder_path)
    for machine_folder in machine_folders:
        mf_path = os.path.join(folder_path, machine_folder)
        if not os.path.isdir(mf_path):
            continue
        files = os.listdir(mf_path)
        files = [f for f in files if f.endswith("_results.json")]
        files = [f for f in files if not f.startswith("-1")]  # remove server in IFCA
        for f in files:
            filepath = os.path.join(mf_path, f)
            with open(filepath, "r") as inf:
                # da = json.load(inf)
                # new_da = {}
                # for k, v in da.items():
                #     if not isinstance(v, dict):
                #         new_da[k] = v
                #         continue
                #     new_da[k] = {kk: vv for kk, vv in v.items() if int(kk) <= 81}
                # results.append(new_da)
                results.append(json.load(inf))
    print("Files", files)

    # data_node = 0
    # with open(folder_path / data_machine / f"{data_node}_results.json", "r") as f:
    #     main_data = json.load(f)
    # main_data = [main_data]

    # Plotting bytes over time
    plt.figure(10)
    b_means, stdevs, mins, maxs = get_stats([x["total_bytes"] for x in results])
    plot(b_means, stdevs, mins, maxs, "Total Bytes", folder_path, "lower right")
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
        os.path.join(folder_path, "total_bytes.csv"),
        index_label="rounds",
    )

    # Plot Training loss
    plt.figure(1)
    means, stdevs, mins, maxs = get_stats([x["train_loss"] for x in results])
    plot(means, stdevs, mins, maxs, "Training Loss", folder_path, "upper right")

    plt.figure(111)
    final_data = get_per_cluster_stats(results, metric="train_loss")
    per_cluster_plot(final_data, "Training Loss per cluster", "upper right")

    # handle the last artificial iteration for all reduce
    if list(iter(b_means.keys())) != list(iter(means.keys())):
        b_means[list(iter(means.keys()))[-1]] = np.nan

    correct_bytes = [b_means[x] for x in means if x in b_means.keys()]
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
    plt.figure(11)
    means = replace_dict_key(means, b_means)
    plot(
        means,
        stdevs,
        mins,
        maxs,
        "Training Loss",
        folder_path,
        "upper right",
        "Total Bytes per node",
    )

    df.to_csv(os.path.join(folder_path, "train_loss.csv"), index_label="rounds")
    # Plot Testing loss
    plt.figure(2)
    means, stdevs, mins, maxs = get_stats([x["test_loss"] for x in results])
    plot(means, stdevs, mins, maxs, "Testing Loss", folder_path, "upper right")
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

    plt.figure(22)
    final_data = get_per_cluster_stats(results, metric="test_loss")
    per_cluster_plot(final_data, "Testing Loss per cluster", "upper right")

    plt.figure(12)
    means = replace_dict_key(means, b_means)
    plot(
        means,
        stdevs,
        mins,
        maxs,
        "Testing Loss",
        folder_path,
        "upper right",
        "Total Bytes per node",
    )

    df.to_csv(os.path.join(folder_path, "test_loss.csv"), index_label="rounds")
    # Plot Testing Accuracy
    plt.figure(3)
    means, stdevs, mins, maxs = get_stats([x["test_acc"] for x in results])
    plot(means, stdevs, mins, maxs, "Testing Accuracy", folder_path, "lower right")
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

    plt.figure(33)
    final_data = get_per_cluster_stats(results, metric="test_acc")
    per_cluster_plot(final_data, "Testing accuracy per cluster", "upper right")

    plt.figure(333, figsize=(10, 6))
    final_data = get_per_cluster_stats(results, metric="test_acc")
    per_cluster_plot_intro(
        final_data, "Testing accuracy for DPSGD (30:2)", "lower right"
    )

    plt.figure(13)
    means = replace_dict_key(means, b_means)
    plot(
        means,
        stdevs,
        mins,
        maxs,
        "Testing Accuracy",
        folder_path,
        "lower right",
        "Total Bytes per node",
    )
    df.to_csv(os.path.join(folder_path, "test_acc.csv"), index_label="rounds")

    # Collect total_bytes shared
    bytes_list = []
    for x in results:
        max_key = str(max(list(map(int, x["total_bytes"].keys()))))
        bytes_list.append({max_key: x["total_bytes"][max_key]})
    means, stdevs, mins, maxs = get_stats(bytes_list)
    bytes_means[folder_path] = list(means.values())[0]
    bytes_stdevs[folder_path] = list(stdevs.values())[0]

    meta_list = []
    for x in results:
        if x["total_meta"]:
            max_key = str(max(list(map(int, x["total_meta"].keys()))))
            meta_list.append({max_key: x["total_meta"][max_key]})
        else:
            meta_list.append({max_key: 0})
    means, stdevs, mins, maxs = get_stats(meta_list)
    meta_means[folder_path] = list(means.values())[0]
    meta_stdevs[folder_path] = list(stdevs.values())[0]

    data_list = []
    for x in results:
        max_key = str(max(list(map(int, x["total_data_per_n"].keys()))))
        data_list.append({max_key: x["total_data_per_n"][max_key]})
    means, stdevs, mins, maxs = get_stats(data_list)
    data_means[folder_path] = list(means.values())[0]
    data_stdevs[folder_path] = list(stdevs.values())[0]

    plt.figure(10)
    plt.savefig(os.path.join(folder_path, "total_bytes.png"), dpi=300)
    plt.figure(11)
    plt.savefig(os.path.join(folder_path, "bytes_train_loss.png"), dpi=300)
    plt.figure(12)
    plt.savefig(os.path.join(folder_path, "bytes_test_loss.png"), dpi=300)
    plt.figure(13)
    plt.savefig(os.path.join(folder_path, "bytes_test_acc.png"), dpi=300)

    plt.figure(1)
    plt.savefig(os.path.join(folder_path, "train_loss.png"), dpi=300)
    plt.figure(111)
    plt.savefig(os.path.join(folder_path, "train_loss_per_cluster.png"), dpi=300)
    plt.figure(2)
    plt.savefig(os.path.join(folder_path, "test_loss.png"), dpi=300)
    plt.figure(22)
    plt.savefig(os.path.join(folder_path, "test_loss_per_cluster.png"), dpi=300)
    plt.figure(3)
    plt.savefig(os.path.join(folder_path, "test_acc.png"), dpi=300)
    plt.figure(33)
    plt.savefig(os.path.join(folder_path, "test_acc_per_cluster.png"), dpi=300)
    plt.figure(333)
    plt.savefig(os.path.join(folder_path, "test_acc_per_cluster_intro.png"), dpi=300)
    plt.savefig(os.path.join(folder_path, "test_acc_per_cluster_intro.pdf"), dpi=300)

    # Plot total_bytes
    plt.figure(4)
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
    plt.figure(5)
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
        plt.figure(6)
        plot_final_cluster_model_attribution(folder_path, results)

        plt.figure(7)
        plot_cluster_model_evolution(folder_path, results)

        plt.figure(8)
        plot_cluster_variation(folder_path, results)

    if "per_sample_loss_train" in results[0].keys():
        # for MIA
        for res in results:
            res["per_sample_loss_train"] = {
                k: json.loads(v) for k, v in res["per_sample_loss_train"].items()
            }
            res["per_sample_loss_test"] = {
                k: json.loads(v) for k, v in res["per_sample_loss_test"].items()
            }
        plt.figure(9)
        plot_loss_distribution(results, folder_path)
        plt.figure(10)
        plot_AUC(results, folder_path)
        plt.figure(11)
        plot_per_cluster_AUC(results, folder_path)

    if (
        "per_sample_pred_test" in results[0].keys()
        and results[0]["per_sample_pred_test"]
    ):
        for res in results:
            res["per_sample_pred_test"] = {
                k: json.loads(v) for k, v in res["per_sample_pred_test"].items()
            }
            res["per_sample_true_test"] = {
                k: json.loads(v) for k, v in res["per_sample_true_test"].items()
            }
        per_class_rates, per_cluster_rates = compute_rates(results)
        plt.figure()
        plot_per_class_demographic_parity(per_class_rates, folder_path)
        plt.figure()
        plot_per_class_equal_opportunities(per_class_rates, folder_path)


def plot_cluster_model_evolution(folder_path, results):
    data = [
        (x["cluster_assigned"], int(k), v)
        for x in results
        for k, v in x["test_best_model_idx"].items()
    ]
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
    max_iter = max(
        [int(iter) for x in results for iter in x["test_best_model_idx"].keys()]
    )

    data = [
        (x["cluster_assigned"], x["test_best_model_idx"][str(max_iter)])
        for x in results
    ]
    df = pd.DataFrame(data, columns=["Cluster Assigned", "Best Model"])
    heatmap_data = (
        df.groupby(["Cluster Assigned", "Best Model"]).size().reset_index(name="Count")
    )
    heatmap_matrix = heatmap_data.pivot(
        index="Cluster Assigned", columns="Best Model", values="Count"
    ).fillna(0)
    plt.figure(figsize=(8, 6))
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


def plot_parameters(path):
    plt.figure(4)
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


def plot_AUC(results, folder_path):
    iterations_to_attack = list(results[0]["per_sample_loss_train"].keys())
    auc_means_iterations, auc_stdev_iterations, counts_iterations = get_auc_means_iter(
        results, iterations_to_attack
    )

    df = pd.DataFrame(
        {
            "Iterations": iterations_to_attack,
            "mean": auc_means_iterations,
            "std": auc_stdev_iterations,
            "Counts": counts_iterations,
        }
    )
    df.to_csv(os.path.join(Path(folder_path), "iterations_threshold.csv"))
    plt.plot(iterations_to_attack, auc_means_iterations)
    plt.fill_between(
        iterations_to_attack,
        auc_means_iterations - auc_stdev_iterations,
        auc_means_iterations + auc_stdev_iterations,
        alpha=0.1,
        lw=2,
    )

    # horizontal line at y = 0.5
    plt.plot(
        [iterations_to_attack[0], iterations_to_attack[-1]],
        [0.5, 0.5],
        color="navy",
        lw=2,
        linestyle="--",
    )

    plt.title("Mean AUC of the LOSS-MIA across all nodes")
    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.xticks([e for i, e in enumerate(iterations_to_attack) if i % 5 == 0])
    # plt.legend(loc="lower right")
    plt.ylim(0.47, 0.7)
    plt.savefig(os.path.join(folder_path, "AUC_iterations.png"))


def plot_per_cluster_AUC(results, folder_path):
    iterations_to_attack = list(results[0]["per_sample_loss_train"].keys())
    clusters = list(set([x["cluster_assigned"] for x in results]))
    clusters.sort()
    num_nodes = len(results)
    minor_exp = False
    for clust in clusters:
        res = [x for x in results if x["cluster_assigned"] == clust]
        if len(res) < num_nodes / len(clusters):
            clust_is_minor = True
            minor_exp = True
            len_minor = len(res)
        else:
            clust_is_minor = False
        (
            auc_means_iterations,
            auc_stdev_iterations,
            counts_iterations,
        ) = get_auc_means_iter(res, iterations_to_attack)

        df = pd.DataFrame(
            {
                "Iterations": iterations_to_attack,
                "mean": auc_means_iterations,
                "std": auc_stdev_iterations,
                "Counts": counts_iterations,
            }
        )
        df.to_csv(
            os.path.join(Path(folder_path), f"iterations_threshold_clust_{clust}.csv")
        )
        if clust_is_minor:
            label = f"Cluster {clust} (Minority)"
        else:
            label = f"Cluster {clust}"
        plt.plot(iterations_to_attack, auc_means_iterations, label=label)
        plt.fill_between(
            iterations_to_attack,
            auc_means_iterations - auc_stdev_iterations,
            auc_means_iterations + auc_stdev_iterations,
            alpha=0.1,
            lw=2,
        )

    # horizontal line at y = 0.5
    plt.plot(
        [iterations_to_attack[0], iterations_to_attack[-1]],
        [0.5, 0.5],
        color="navy",
        lw=2,
        linestyle="--",
    )

    if minor_exp:
        ratio = f"\nRatio: {num_nodes-len_minor}:{len_minor}"
    else:
        ratio = ""
    plt.title(f"Mean AUC of the LOSS-MIA across all nodes of the same cluster{ratio}")
    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.xticks([e for i, e in enumerate(iterations_to_attack) if i % 5 == 0])
    plt.legend(loc="lower right")
    plt.ylim(0.47, 0.7)
    plt.savefig(os.path.join(folder_path, "AUC_iterations_per_cluster.png"))


def plot_loss_distribution(results, folder_path):
    for res in results:
        iterations = list(res["per_sample_loss_train"].keys())
        train_count = len(res["per_sample_loss_train"][iterations[0]])
        test_count = len(res["per_sample_loss_test"][iterations[0]])
        smallest_count = min(train_count, test_count)
        for (iter, tr_data), te_data in zip(
            res["per_sample_loss_train"].items(), res["per_sample_loss_test"].values()
        ):
            tr_data_trunc = tr_data[:smallest_count]
            te_data_trunc = te_data[:smallest_count]
            if iter == "79":
                plt.hist(tr_data_trunc, bins=100, alpha=0.5, label="train")
                plt.hist(te_data_trunc, bins=100, alpha=0.5, label="test")
                plt.title("Not Finished, do not show")
                plt.savefig(os.path.join(folder_path, "train_loss_distribution.png"))
        break


def get_auc_means_iter(results, iterations_to_attack):
    auc_iter = []
    for data_per_node in results:
        aucs = []
        for (iter, tr_data), te_data in zip(
            data_per_node["per_sample_loss_train"].items(),
            data_per_node["per_sample_loss_test"].values(),
        ):
            if iter in iterations_to_attack:
                _, _, _, roc_auc = get_roc_auc(
                    tr_data,
                    te_data,
                )
                aucs.append(roc_auc)
        auc_iter.append(aucs)
    counts = np.array([len(auc_iter)] * len(auc_iter[0]))
    auc_means_iter = np.mean(auc_iter, axis=0)
    auc_stdev_iter = np.std(auc_iter, axis=0)

    return auc_means_iter, auc_stdev_iter, counts


def get_roc_auc(tr_data, te_data):
    # Get the ROC AUC for the given iteration
    train_count = len(tr_data)
    test_count = len(te_data)
    smallest_count = min(train_count, test_count)
    tr_data_minus = -np.array(tr_data[:smallest_count])
    te_data_minus = -np.array(te_data[:smallest_count])
    tr_data_true = np.ones((smallest_count,), dtype=np.int32)
    te_data_true = np.zeros((smallest_count,), dtype=np.int32)
    y_true = np.concatenate((tr_data_true, te_data_true))
    y_loss_minus = np.concatenate((tr_data_minus, te_data_minus))

    fpr, tpr, thresholds = roc_curve(y_true, y_loss_minus)
    roc_auc = roc_auc_score(y_true, y_loss_minus)
    return fpr, tpr, thresholds, roc_auc


def compute_rates(results: List[Dict[str, Any]]):
    """Computes the rates across all nodes for each clusters and each class.
    The numbers are summed across all nodes and all classes (we don't use the mean).

    Args:
        results (List): _description_

    Returns:
        _type_: _description_
    """
    clusters = set([int(x["cluster_assigned"]) for x in results])
    per_cluster_rates = {x: {"TP": [], "TN": [], "FP": [], "FN": []} for x in clusters}
    for res in results:
        cluster = res["cluster_assigned"]
        # in recent data, we only record the last iteration to save time
        if not res["per_sample_pred_test"]:
            return None, None  # exp was manually stop, not have the correct data
        last_iter = str(max([int(x) for x in res["per_sample_pred_test"].keys()]))
        for (iter, pred), (_, true) in zip(
            res["per_sample_pred_test"].items(), res["per_sample_true_test"].items()
        ):
            # for now only care about the last iteration
            if iter == last_iter:
                tp, tn, fp, fn = per_class_perf_measure(true, pred)
                per_cluster_rates[cluster]["TP"].append(tp)
                per_cluster_rates[cluster]["TN"].append(tn)
                per_cluster_rates[cluster]["FP"].append(fp)
                per_cluster_rates[cluster]["FN"].append(fn)
    per_class_rates = {
        k: {kk: np.mean(vv, axis=0) for kk, vv in v.items()}
        for k, v in per_cluster_rates.items()
    }
    per_cluster_rates = {
        k: {kk: np.sum(vv) for kk, vv in v.items()} for k, v in per_class_rates.items()
    }
    return per_class_rates, per_cluster_rates


def per_class_perf_measure(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = np.array((cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)), dtype=np.float32)
    FN = np.array((cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)), dtype=np.float32)
    TP = np.array((np.diag(cnf_matrix)).astype(float), dtype=np.float32)
    TN = np.array((cnf_matrix.sum() - (FP + FN + TP)).astype(float), dtype=np.float32)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)

    return TP, TN, FP, FN


def plot_per_class_demographic_parity(
    per_class_rates: Dict[int, Dict[str, int]], folder_path
):
    """Compute and plot the demographic parity with S the sensitive attribute beeing the cluster belonging of each node.
        TP + FP / all preds
    Args:
        per_class_rates (Dict): _description_
        folder_path (_type_): _description_
    """
    clusters = list(per_class_rates.keys())
    pos_preds_0 = (
        per_class_rates[clusters[0]]["TP"] + per_class_rates[clusters[0]]["FP"]
    )
    tot_0 = np.sum(
        [per_class_rates[clusters[0]][k] for k in per_class_rates[clusters[0]].keys()],
        axis=0,
    )
    pos_preds_1 = (
        per_class_rates[clusters[1]]["TP"] + per_class_rates[clusters[1]]["FP"]
    )
    tot_1 = np.sum(
        [per_class_rates[clusters[1]][k] for k in per_class_rates[clusters[1]].keys()],
        axis=0,
    )

    demo_parity = abs(pos_preds_0 / tot_0 - pos_preds_1 / tot_1)

    df = pd.DataFrame(
        {
            "class_labels": [CIFAR_CLASSES[i] for i in range(len(demo_parity))],
            "demographic_parity": demo_parity,
        }
    )
    df.to_csv(os.path.join(folder_path, "demographic_parity.csv"))

    plt.plot(demo_parity, "o")
    plt.title("Per class demographic parity")
    plt.ylabel("Absolute difference in accuracy")
    plt.xlabel("Classes")
    plt.xticks(
        range(len(demo_parity)),
        [CIFAR_CLASSES[i] for i in range(len(demo_parity))],
        rotation=45,
    )
    plt.savefig(os.path.join(folder_path, "demographic_parity.png"), dpi=300)


def plot_per_class_equal_opportunities(
    per_class_rates: Dict[int, Dict[str, int]], folder_path
):
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

    df = pd.DataFrame(
        {
            "class_labels": [CIFAR_CLASSES[i] for i in range(len(eq_op))],
            "equal_opportunities": eq_op,
        }
    )
    df.to_csv(os.path.join(folder_path, "equal_opportunities.csv"))

    plt.plot(eq_op, "o")
    plt.title("Per class equal opportunities")
    plt.ylabel("Absolute difference in recall")
    plt.xlabel("Classes")
    plt.xticks(
        range(len(eq_op)), [CIFAR_CLASSES[i] for i in range(len(eq_op))], rotation=45
    )
    plt.savefig(os.path.join(folder_path, "equal_opportunities.png"), dpi=300)


def per_class_equalized_odds(per_class_rates: Dict[int, Dict[str, int]], folder_path):
    """Compute and plot the equalized odds with S the sensitive attribute beeing the cluster belonging of each node.
        (TP + ) ??

    Args:
        per_class_rates (Dict[int, Dict[str, int]]): _description_
        folder_path (_type_): _description_
    """
    pass


if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: the folder with the data
    plot_results(sys.argv[1])
    # plot_parameters(sys.argv[1])

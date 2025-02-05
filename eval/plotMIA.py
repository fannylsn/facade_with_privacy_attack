"""Obsolete, use plotIDCA.py instead"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_results(folder_path, data_machine="machine0", data_node=0):
    print("Reading the folder: ", folder_path)
    folder_path = Path(os.path.abspath(folder_path))

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
                results.append(json.load(inf))
    print("Files", files)
    for res in results:
        res["per_sample_loss_train"] = {
            k: json.loads(v) for k, v in res["per_sample_loss_train"].items()
        }
        res["per_sample_loss_test"] = {
            k: json.loads(v) for k, v in res["per_sample_loss_test"].items()
        }
    plt.figure(1)
    plot_loss_distribution(results, folder_path)
    plt.figure(2)
    plot_AUC(results, folder_path)
    plt.figure(3)
    plot_per_cluster_AUC(results, folder_path)


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


if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: the folder with the data
    plot_results(sys.argv[1])
    # plot_parameters(sys.argv[1])

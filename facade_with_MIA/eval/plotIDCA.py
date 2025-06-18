import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Constants for class labels
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

FLICKR_CLASSES = {k: str(k) for k in range(41)}
DATA_CLASSES = FLICKR_CLASSES

# Plot settings
plt.rcParams.update({"font.size": 16})
plt.rc("legend", fontsize=11)

# Cluster names for plotting
CLUSTER_NAMES = {0: "A", 1: "B", 2: "C", 3: "D"}

def get_stats(data: List[Dict[str, Any]]) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Calculate statistics (mean, standard deviation, min, max) for each key across a list of dictionaries.

    Args:
        data: List of dictionaries containing numerical data.

    Returns:
        Tuple of dictionaries containing mean, standard deviation, min, and max for each key.
    """
    assert len(data) > 0
    mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}

    for key in data[0].keys():
        all_nodes = np.array([i[key] for i in data])
        mean_dict[int(key)] = np.mean(all_nodes)
        stdev_dict[int(key)] = np.std(all_nodes)
        min_dict[int(key)] = np.min(all_nodes)
        max_dict[int(key)] = np.max(all_nodes)

    return mean_dict, stdev_dict, min_dict, max_dict

def get_per_cluster_stats(results: List[Dict[str, Any]], metric: str = "test_loss") -> Dict[int, List[Dict[int, float]]]:
    """
    Calculate statistics per cluster for a given metric.

    Args:
        results: List of result dictionaries.
        metric: Metric to calculate statistics for.

    Returns:
        Dictionary containing statistics for each cluster.
    """
    assert len(results) > 0
    clusters = set([int(x["cluster_assigned"]) for x in results])
    final_data = {}

    for clust in clusters:
        mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}
        for key in results[0][metric].keys():
            all_nodes = np.array([i[metric][key] for i in results if i["cluster_assigned"] == clust])
            mean_dict[int(key)] = np.mean(all_nodes)
            stdev_dict[int(key)] = np.std(all_nodes)
            min_dict[int(key)] = np.min(all_nodes)
            max_dict[int(key)] = np.max(all_nodes)

        final_data[clust] = [mean_dict, stdev_dict, min_dict, max_dict]

    return final_data

def unified_plot(
    results: Optional[List[Dict[str, Any]]] = None,
    means: Optional[Dict[int, float]] = None,
    stdevs: Optional[Dict[int, float]] = None,
    title: str = "",
    label: Optional[str] = None,
    loc: str = "lower right",
    xlabel: str = "communication rounds",
    ylabel: str = "",
    yticks: Optional[List[float]] = None,
    metric: str = "",
    return_data: bool = False,
    cluster_assign: Optional[List[int]] = None,
) -> Optional[Dict[int, List[Dict[int, float]]]]:
    """
    Unified function to plot data with error bars or per cluster statistics.

    Args:
        results: List of result dictionaries for per-cluster plotting.
        means: Dictionary of mean values for plotting.
        stdevs: Dictionary of standard deviations for plotting.
        mins: Dictionary of minimum values.
        maxs: Dictionary of maximum values.
        title: Title of the plot.
        label: Label for the plot.
        loc: Location of the legend.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        yticks: Ticks for the y-axis.
        metric: Metric to calculate statistics for.
        return_data: Whether to return the final data.
        cluster_assign: List of cluster assignments for per-cluster plotting.

    Returns:
        Optional dictionary containing statistics for each cluster if return_data is True.
    """
    if results is not None:
        final_data = get_per_cluster_stats(results, metric=metric)
        clusters_idxs = set(cluster_assign)
        clust_count = {clust: cluster_assign.count(clust) for clust in clusters_idxs}

        for clust, data in final_data.items():
            means, stdevs, _, _ = data
            x_axis = np.array(list(means.keys()))
            y_axis = np.array(list(means.values()))
            err = np.array(list(stdevs.values()))
            minority = clust_count[clust] < clust_count[(clust + 1) % 2]
            label = f"cluster {CLUSTER_NAMES[clust]} (minority)" if minority else f"cluster {CLUSTER_NAMES[clust]}"
            plt.plot(x_axis, y_axis, label=label)
            plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)

        if yticks is not None:
            plt.yticks(yticks, fontsize=10)
            plt.ylim(0, yticks[-1])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, color="gray", linestyle="-", linewidth=0.5)
        plt.title(title + f" (ratio {clust_count[0]}:{clust_count[1]})", fontsize=16)
        plt.legend(loc=loc, ncols=2)
        plt.tight_layout()

        if return_data:
            return final_data

    else:
        x_axis = np.array(list(means.keys()))
        y_axis = np.array(list(means.values()))
        err = np.array(list(stdevs.values()))

        plt.plot(x_axis, y_axis, label=label)
        plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)

        if yticks is not None:
            plt.yticks(yticks)
            plt.ylim(0, yticks[-1])

        if label is not None:
            plt.legend(loc=loc)

        plt.title(title, fontdict={"fontsize": 18})
        plt.xlabel(xlabel, fontdict={"fontsize": 12})
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, color="gray", linestyle="-", linewidth=0.5)
        plt.tight_layout()


def replace_dict_key(d_org: Dict[Any, Any], d_other: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Replace keys in a dictionary using a mapping from another dictionary.

    Args:
        d_org: Original dictionary.
        d_other: Dictionary with key mappings.

    Returns:
        Dictionary with replaced keys.
    """
    return {d_other[x]: y for x, y in d_org.items()}


def plot_cluster_model_evolution(folder_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Plot the evolution of cluster models over time.

    Args:
        folder_path: Path to the folder where the plot will be saved.
        results: List of result dictionaries.
    """
    if results[0]["train_best_model_idx"] == {}:
        data = []
        for res in results:
            all_train_loss = {}
            for outer_key, inner_dict in res["all_train_loss"].items():
                for inner_key, value in inner_dict.items():
                    if inner_key not in all_train_loss:
                        all_train_loss[inner_key] = {}
                    all_train_loss[inner_key][outer_key] = value
            train_best_idx = [(res["cluster_assigned"], int(k), list(v.values()).index(best_loss))
                              for (k, v), best_loss in zip(all_train_loss.items(), res["train_loss"].values())]
            data.extend(train_best_idx)
    else:
        data = [(x["cluster_assigned"], int(k), v)
                for x in results
                for k, v in x["train_best_model_idx"].items()]

    clusters = set([x[0] for x in data])
    models = set([x[2] for x in data])
    max_iter = max([el[1] for el in data])
    space_iter = data[1][1] - data[0][1]
    if space_iter == 0:
        space_iter = 1
    init_iter = data[0][1]
    fig, axs = plt.subplots(len(clusters), 1, figsize=(10, 9))
    fig.suptitle("Number of nodes in each cluster picking each head")

    for i, cluster in enumerate(clusters):
        cluster_data = [x for x in data if x[0] == cluster]
        for model in models:
            model_data = [x for x in cluster_data if x[2] == model]
            count = sorted(Counter(model_data).items(), key=lambda pair: pair[0][1])
            for j in range(init_iter, max_iter + 1, space_iter):
                if j not in [elem[0][1] for elem in count]:
                    count.append(((cluster, j, model), 0))
            count = sorted(count, key=lambda pair: pair[0][1])
            axs[i].plot([elem[0][1] for elem in count], [elem[1] for elem in count], label=f"model {model}")

        axs[i].set_ylim(ymin=0)
        axs[i].set_xticks(np.arange(0, max_iter + 1, 10))
        axs[i].title.set_text(f"{int(len(cluster_data)/len(count))} nodes in cluster {CLUSTER_NAMES[cluster]}")
        axs[i].legend(loc="upper right", fontsize="8")
        axs[i].set_ylabel("# of nodes")
        axs[i].set_xlabel("Communication Rounds")

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "cluster_model_evolution.png"), dpi=300)
    plt.close()


def plot_final_cluster_model_attribution(folder_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Plot the final cluster model attribution.

    Args:
        folder_path: Path to the folder where the plot will be saved.
        results: List of result dictionaries.
    """
    if results[0]["test_best_model_idx"] == {}:
        plot_final_train_cluster_model_attribution(folder_path, results)
        return

    max_iter = max([int(iter) for x in results for iter in x["test_best_model_idx"].keys()])

    data = [(x["cluster_assigned"], x["test_best_model_idx"][str(max_iter)]) for x in results]
    df = pd.DataFrame(data, columns=["Cluster Assigned", "Best Head"])
    heatmap_data = df.groupby(["Cluster Assigned", "Best Head"]).size().reset_index(name="Count")
    heatmap_matrix = heatmap_data.pivot(index="Cluster Assigned", columns="Best Head", values="Count").fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt="g", cmap="YlGnBu")
    plt.title("Final Head choice")
    plt.savefig(os.path.join(folder_path, "cluster_model_distribution.png"), dpi=300)
    plt.close()


def plot_final_train_cluster_model_attribution(folder_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Plot the final train cluster model attribution.

    Args:
        folder_path: Path to the folder where the plot will be saved.
        results: List of result dictionaries.
    """
    data = []
    for res in results:
        all_train_loss = {}
        for outer_key, inner_dict in res["all_train_loss"].items():
            for inner_key, value in inner_dict.items():
                if inner_key not in all_train_loss:
                    all_train_loss[inner_key] = {}
                all_train_loss[inner_key][outer_key] = value

        last_iter = str(max([int(k) for k in all_train_loss.keys()]))
        final_best_idx = (res["cluster_assigned"], list(all_train_loss[last_iter].values()).index(res["train_loss"][last_iter]))
        data.append(final_best_idx)

    df = pd.DataFrame(data, columns=["Cluster Assigned", "Best Model"])
    heatmap_data = df.groupby(["Cluster Assigned", "Best Model"]).size().reset_index(name="Count")
    heatmap_matrix = heatmap_data.pivot(index="Cluster Assigned", columns="Best Model", values="Count").fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, annot=True, fmt="g", cmap="YlGnBu")
    plt.title("Heatmap of Data Distribution")
    plt.savefig(os.path.join(folder_path, "cluster_model_distribution.png"), dpi=300)
    plt.close()


def plot_all_models_train_loss(folder_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Plot the training loss of all models on the node's dataset averaged across all nodes in each cluster.

    Args:
        folder_path: Path to the folder where the plot will be saved.
        results: List of result dictionaries.
    """
    real_cluster = [x["cluster_assigned"] for x in results]
    fig, axs = plt.subplots(1, len(set(real_cluster)))
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.suptitle("Training loss of each models on the node's dataset\naveraged across all nodes in each cluster")

    for clust in sorted(list(set(real_cluster))):
        data = [x for x in results if x["cluster_assigned"] == clust]
        all_models = sorted(list(set(data[0]["all_train_loss"].keys())))

        for model in all_models:
            model_data = [x["all_train_loss"][model] for x in data]
            means, stdevs, _, _ = get_stats(model_data)
            x = np.array(list(means.keys()))
            y = np.array(list(means.values()))
            yerr = np.array(list(stdevs.values()))

            axs[clust].plot(x, y, label=f"model {model}")
            axs[clust].fill_between(x, y - yerr, y + yerr, alpha=0.4)

        axs[clust].set_title(f"Cluster {CLUSTER_NAMES[clust]}\n({len(data)} nodes)")
        axs[clust].set_xlabel("Comm. rounds")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    axs[0].set_ylabel("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "all_models_train_loss.png"), dpi=300)
    plt.close()


def plot_fair_accuracy(final_data: Dict[int, List[Dict[int, float]]], folder_path: str, label: str = "Fair Accuracy", mean_type: str = "weighted") -> Dict[int, float]:
    """
    Plot the fair accuracy metric.

    Args:
        final_data: Dictionary containing data for each cluster.
        folder_path: Path to the folder where the plot will be saved.
        label: Label for the plot.
        mean_type: Type of mean to use for the fair accuracy calculation.

    Returns:
        Dictionary containing the fair accuracy values.
    """
    mean_acc = {}
    acc_similarity = {}
    fair_accuracy = {}

    for it in final_data[0][0]:
        accs = [acc[0][it] for acc in final_data.values()]
        mean_acc[it] = np.mean(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        acc_similarity[it] = 100 - (max_acc - min_acc)

        if mean_type == "arithmetic":
            fair_accuracy[it] = (mean_acc[it] + acc_similarity[it]) / 2
        elif mean_type == "harmonic":
            fair_accuracy[it] = 2 * mean_acc[it] * acc_similarity[it] / (mean_acc[it] + acc_similarity[it])
        elif mean_type == "weighted":
            fair_accuracy[it] = (2 * mean_acc[it] + acc_similarity[it]) / 3

    plt.plot(list(fair_accuracy.keys()), list(fair_accuracy.values()), label=label)
    plt.title("Fair Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel(f"{mean_type} mean of mean accuracy and accuracy similarity")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "fair_acc.png"))
    return fair_accuracy


def compute_rates(results: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, int]]]:
    """
    Compute the rates across all nodes for each cluster and each class.

    Args:
        results: List of result dictionaries.

    Returns:
        Tuple of dictionaries containing per class rates and per cluster rates.
    """
    clusters = set([int(x["cluster_assigned"]) for x in results])
    per_cluster_rates = {x: {"TP": [], "TN": [], "FP": [], "FN": []} for x in clusters}

    for res in results:
        cluster = res["cluster_assigned"]
        if not res["per_sample_pred_test"]:
            return None, None

        last_iter = str(max([int(x) for x in res["per_sample_pred_test"].keys()]))

        for (iter, pred), (_, true) in zip(res["per_sample_pred_test"].items(), res["per_sample_true_test"].items()):
            if iter == last_iter:
                tp, tn, fp, fn = per_class_perf_measure(true, pred)
                per_cluster_rates[cluster]["TP"].append(tp)
                per_cluster_rates[cluster]["TN"].append(tn)
                per_cluster_rates[cluster]["FP"].append(fp)
                per_cluster_rates[cluster]["FN"].append(fn)

    per_class_rates = {k: {kk: np.mean(vv, axis=0) for kk, vv in v.items()} for k, v in per_cluster_rates.items()}
    per_cluster_rates = {k: {kk: np.sum(vv) for kk, vv in v.items()} for k, v in per_class_rates.items()}

    return per_class_rates, per_cluster_rates


def per_class_perf_measure(y_true: List[int], y_pred: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute performance measures for each class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Tuple of arrays containing true positives, true negatives, false positives, and false negatives.
    """
    all_labels = list(set(y_true))
    cnf_matrix = compute_confusion_matrix(y_true, y_pred, all_labels)

    FP = np.array((cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)), dtype=np.float32)
    FN = np.array((cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)), dtype=np.float32)
    TP = np.array((np.diag(cnf_matrix)).astype(float), dtype=np.float32)
    TN = np.array((cnf_matrix.sum() - (FP + FN + TP)).astype(float), dtype=np.float32)

    return TP, TN, FP, FN


def compute_confusion_matrix(y_true: List[int], y_pred_topk: List[int], labels: List[int]) -> np.ndarray:
    """
    Compute the confusion matrix for given true and predicted labels.

    Args:
        y_true: True labels.
        y_pred_topk: Predicted labels.
        labels: List of unique labels.

    Returns:
        Confusion matrix.
    """
    num_classes = len(labels)
    cnf_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_classes = y_pred_topk[i]

        if isinstance(pred_classes, int):
            pred_classes = [pred_classes]

        if true_class in pred_classes:
            cnf_matrix[true_class, true_class] += 1

        for pred_class in pred_classes:
            if pred_class != true_class:
                cnf_matrix[true_class, pred_class] += 1

    return cnf_matrix


def plot_per_class_demographic_parity(per_class_rates: Dict[int, Dict[str, np.ndarray]], folder_path: str, config: Optional[str] = None) -> None:
    """
    Compute and plot the demographic parity with S the sensitive attribute being the cluster belonging of each node.

    Args:
        per_class_rates: Dictionary containing per class rates.
        folder_path: Path to the folder where the plot will be saved.
        config: Configuration label for the plot.
    """
    clusters = list(per_class_rates.keys())
    pos_preds_0 = per_class_rates[clusters[0]]["TP"] + per_class_rates[clusters[0]]["FP"]
    tot_0 = np.sum([per_class_rates[clusters[0]][k] for k in per_class_rates[clusters[0]].keys()], axis=0)

    pos_preds_1 = per_class_rates[clusters[1]]["TP"] + per_class_rates[clusters[1]]["FP"]
    tot_1 = np.sum([per_class_rates[clusters[1]][k] for k in per_class_rates[clusters[1]].keys()], axis=0)

    demo_parity = abs(pos_preds_0 / tot_0 - pos_preds_1 / tot_1)
    all = np.mean(demo_parity)
    labels = [DATA_CLASSES[i] for i in range(len(demo_parity))]
    labels.append("Mean")
    with_mean = np.append(demo_parity, all)

    df = pd.DataFrame({"class_labels": labels, "demographic_parity": with_mean})
    df.to_csv(os.path.join(folder_path, "demographic_parity.csv"))

    if config is None:
        plt.plot(with_mean, "o")
    else:
        plt.plot(with_mean, "o", label=config)
        plt.legend(loc="upper right")

    plt.title("Per class demographic parity")
    plt.ylabel("Absolute difference in accuracy")
    plt.xlabel("Classes")
    plt.xticks(range(len(with_mean)), labels, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "demographic_parity.png"), dpi=300)


def per_class_equalized_odds(per_class_rates: Dict[int, Dict[str, np.ndarray]], folder_path: str, config: Optional[str] = None) -> None:
    """
    Compute and plot the equalized odds with S the sensitive attribute being the cluster belonging of each node.

    Args:
        per_class_rates: Dictionary containing per class rates.
        folder_path: Path to the folder where the plot will be saved.
        config: Configuration label for the plot.
    """
    clusters = list(per_class_rates.keys())
    rec_0 = per_class_rates[clusters[0]]["TP"] / (per_class_rates[clusters[0]]["TP"] + per_class_rates[clusters[0]]["FN"])
    rec_1 = per_class_rates[clusters[1]]["TP"] / (per_class_rates[clusters[1]]["TP"] + per_class_rates[clusters[1]]["FN"])
    equal_oppo = abs(rec_0 - rec_1)

    fpr_0 = per_class_rates[clusters[0]]["FP"] / (per_class_rates[clusters[0]]["FP"] + per_class_rates[clusters[0]]["TN"])
    fpr_1 = per_class_rates[clusters[1]]["FP"] / (per_class_rates[clusters[1]]["FP"] + per_class_rates[clusters[1]]["TN"])
    fpr_diff = abs(fpr_0 - fpr_1)

    eq_odds = equal_oppo + fpr_diff

    all = np.mean(eq_odds)
    labels = [DATA_CLASSES[i] for i in range(len(eq_odds))]
    labels.append("Mean")
    eq_odds = np.append(eq_odds, all)

    df = pd.DataFrame({"class_labels": labels, "equalized_odds": eq_odds})
    df.to_csv(os.path.join(folder_path, "equalized_odds.csv"))

    if config is None:
        plt.plot(eq_odds, "o")
    else:
        plt.plot(eq_odds, "o", label=config)
        plt.legend(loc="upper right")

    plt.title("Per class equalized odds")
    plt.ylabel("Absolute difference of recall and false positive rate")
    plt.xlabel("Classes")
    plt.xticks(range(len(eq_odds)), labels, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "equalized_odds.png"), dpi=300)


def plot_results(folder_path: str) -> None:
    """
    Plot results from a folder containing JSON files with data.

    Args:
        folder_path: Path to the folder containing data.
    """
    print("Reading the folder: ", folder_path)
    folder_path = Path(os.path.abspath(folder_path))

    results = []
    machine_folders = os.listdir(folder_path)

    for machine_folder in machine_folders:
        mf_path = os.path.join(folder_path, machine_folder)
        if not os.path.isdir(mf_path):
            continue

        files = [f for f in os.listdir(mf_path) if f.endswith("_results.json") and not f.startswith("-1")]
        files = sorted(files, key=lambda x: int(x.split('_')[0]))

        for f in files:
            filepath = os.path.join(mf_path, f)
            with open(filepath, "r") as inf:
                results.append(json.load(inf))

    b_means, stdevs, mins, maxs = get_stats([x["total_bytes"] for x in results])

    plt.figure(1)
    means, stdevs, _, _ = get_stats([x["train_loss"] for x in results])
    unified_plot(means=means, stdevs=stdevs, title="Training Loss", label="FACADE", loc="upper right")

    plt.figure(111)
    cluster_assign = [x["cluster_assigned"] for x in results]
    unified_plot(results=results, metric="train_loss", title="Training Loss per cluster", loc="upper right", ylabel="Training Loss", cluster_assign=cluster_assign)

    if list(b_means.keys()) != list(means.keys()):
        b_means[list(means.keys())[-1]] = np.nan

    plt.figure(11)
    means = replace_dict_key(means, b_means)
    unified_plot(means=means, stdevs=stdevs, title="Training Loss", label=folder_path, loc="upper right", xlabel="Total Bytes per node")

    plt.figure(2)
    means, stdevs, _, _ = get_stats([x["test_loss"] for x in results])
    unified_plot(means=means, stdevs=stdevs, title="Testing Loss", label="FACADE", loc="upper right")

    df = pd.DataFrame({"mean": list(means.values()), "std": list(stdevs.values()), "nr_nodes": [len(results)] * len(means)}, index=list(means.keys()), columns=["mean", "std", "nr_nodes"])
    df.to_csv(os.path.join(folder_path, "test_loss.csv"), index_label="rounds")

    plt.figure(22)
    unified_plot(results=results, metric="test_loss", title="Testing Loss per cluster", loc="upper right", ylabel="Testing Loss", cluster_assign=cluster_assign)

    plt.figure(12)
    means = replace_dict_key(means, b_means)
    unified_plot(means=means, stdevs=stdevs, title="Testing Loss", label=folder_path, loc="upper right", xlabel="Total Bytes per node")

    plt.figure(3, figsize=(8, 7))
    means, stdevs, _, _ = get_stats([x["test_acc"] for x in results])
    unified_plot(means=means, stdevs=stdevs, title="Testing Accuracy", loc="lower right", yticks=np.arange(0, 81, 10))

    df = pd.DataFrame({"mean": list(means.values()), "std": list(stdevs.values()), "nr_nodes": [len(results)] * len(means)}, index=list(means.keys()), columns=["mean", "std", "nr_nodes"])
    df.to_csv(os.path.join(folder_path, "test_acc.csv"), index_label="rounds")

    plt.figure(13)
    means = replace_dict_key(means, b_means)
    unified_plot(means=means, stdevs=stdevs, title="Testing Accuracy", label=folder_path, loc="lower right", xlabel="Total Bytes per node")

    plt.figure(33)
    final_data = unified_plot(results=results, metric="test_acc", title="Testing accuracy per cluster", loc="lower right", ylabel="Testing Accuracy (%)", yticks=np.arange(0, 81, 10), return_data=True, cluster_assign=cluster_assign)
    df = pd.DataFrame({"mean_0": list(final_data[0][0].values()), "std_0": list(final_data[0][1].values()), "mean_1": list(final_data[1][0].values()), "std_1": list(final_data[1][1].values()), "nr_nodes": [len(results)] * len(means)}, index=list(means.keys()), columns=["mean_0", "std_0", "mean_1", "std_1", "nr_nodes"])
    df.to_csv(os.path.join(folder_path, "test_acc_clut.csv"), index_label="rounds")

    plt.figure(333)
    plot_fair_accuracy(final_data, folder_path)

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

    if "test_best_model_idx" in results[0].keys():
        plt.figure(6)
        plot_final_cluster_model_attribution(folder_path, results)

        plt.figure(7)
        plot_cluster_model_evolution(folder_path, results)

        plt.figure(80)
        plot_all_models_train_loss(folder_path, results)

    if "per_sample_pred_test" in results[0].keys() and results[0]["per_sample_pred_test"]:
        for res in results:
            for k, v in res["per_sample_pred_test"].items():
                if isinstance(v, list):
                    res["per_sample_pred_test"][k] = v
                else:
                    res["per_sample_pred_test"][k] = json.loads(v)
            for k, v in res["per_sample_true_test"].items():
                if isinstance(v, list):
                    res["per_sample_true_test"][k] = v
                else:
                    res["per_sample_true_test"][k] = json.loads(v)
        per_class_rates, _ = compute_rates(results)
        plt.figure()
        plot_per_class_demographic_parity(per_class_rates, folder_path)
        plt.figure()
        per_class_equalized_odds(per_class_rates, folder_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_results(sys.argv[1])

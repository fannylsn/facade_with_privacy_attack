import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/eval/")
from plotIDCA import (
    compute_rates,
)

CONFIGS = {
    "config1": "10:10",
    "config2": "15:5",
    "config3": "18:2",
}

NODE_TYPES = ["IDCA", "IFCA_RESTARTS", "IFCA", "DPSGD", "DEPRL", "DAC"]
COLORS = {"IDCA": "b", "IFCA": "orange", "IFCA_RESTARTS": "darkorange", "DPSGD": "g", "DEPRL": "r", "DAC": "purple"}

LINE_STYLES = ["-", "--"]


def plot_results(parent_folder_path, data_machine="machine0", data_node=0):
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
        for name in NODE_TYPES:
            if name.lower() in str(subdir).lower():
                node_type = name
                break
        print("Reading the folder: ", folder_path)

        all_results, has_multi_model = get_data_from_exp(folder_path)
        all_exp_results[node_type] = all_results
        multi_model[node_type] = has_multi_model
        exp_subdirs[node_type] = subdir

    exp_subdirs = OrderedDict((key, exp_subdirs[key]) for key in NODE_TYPES if key in exp_subdirs)

    # clean in case of missing data
    common_configs = set(all_exp_results[next(iter(all_exp_results))].keys())
    for exp in all_exp_results.values():
        common_configs &= set(exp.keys())
    all_exp_results = {k: {kk: vv for kk, vv in v.items() if kk in common_configs} for k, v in all_exp_results.items()}
    # compute and plot fair metrics
    fair_metrics = get_fair_metrics(all_exp_results)
    plot_fair_metric(fair_metrics, parent_folder_path)

    # general plot
    # plot_all_accs(all_exp_results, parent_folder_path)

    # plotting across iterations
    fig_acc, axs_acc = plt.subplots(len(common_configs), 1, figsize=(10, 15))
    fig_acc.suptitle("Accuracy by config for each experiment")
    fig_acc_clust, axs_acc_clust = plt.subplots(len(common_configs), 1, figsize=(10, 15))
    fig_acc_clust.suptitle("Accuracy by config for each experiment detailed by cluster")
    for exp, all_results in all_exp_results.items():
        acc_per_config, acc_per_config_per_cluster = extract_acc_stats(all_results)
        plt.close()
        plt.figure(fig_acc)
        plot_acc_per_exp(parent_folder_path, acc_per_config, exp, axs_acc)
        plt.close()
        plt.figure(fig_acc_clust)
        plot_acc_per_cluster_per_exp(parent_folder_path, acc_per_config_per_cluster, exp, axs_acc_clust)
        plt.close()
        # if has_multi_model:
        #     plt.figure(2)
        #     plot_settling_time(parent_folder_path, all_results)

        # for i, (config_folder, data) in enumerate(all_results.items()):
        #     for j, (seed_folder, seed_data) in enumerate(data.items()):
        #         exp_path = os.path.join(parent_folder_path, exp_subdirs[exp], config_folder, seed_folder)
        #         if multi_model[exp]:
        #             plot_cluster_model_evolution(exp_path, seed_data)
        #             plot_cluster_variation(exp_path, seed_data)
        #             plot_final_cluster_model_attribution(exp_path, seed_data)


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

            mf_path = os.path.join(seed_folder_path, "machine0")

            files = os.listdir(mf_path)
            files = [f for f in files if f.endswith("_results.json")]
            files = [f for f in files if not f.startswith("-1")]  # remove server in IFCA
            for f in files:
                filepath = os.path.join(mf_path, f)
                with open(filepath, "r") as inf:
                    results.append(json.load(inf))
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
                    res["per_sample_pred_test"] = {k: json.loads(v) for k, v in res["per_sample_pred_test"].items()}
                    res["per_sample_true_test"] = {k: json.loads(v) for k, v in res["per_sample_true_test"].items()}
            for metric in METRIC_FUNCS.keys():
                temp_metric = []
                # iterate on all same exp with different seeds
                for results in data.values():
                    # now results is the results of one exp, with config config
                    per_class_rates, per_cluster_rates = compute_rates(results)
                    if per_class_rates is None:
                        continue
                    temp_metric.append(METRIC_FUNCS[metric](per_cluster_rates, per_class_rates))  # call to correct func

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

        plt.close()
        plt.figure()
        for node_type, vals, std in zip(node_types, values, std_dev):
            plt.errorbar(index, vals, std, fmt="o", linewidth=2, capsize=6, label=node_type, color=COLORS[node_type])

        plt.xlabel("Ratio of majority (cluster 0) to minority (cluster 1)")
        plt.ylabel(metric)
        plt.title(f"{metric} by experiment")
        plt.xticks(index, map(lambda x: CONFIGS[x], configs))
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{metric}_per_node_type_per_config.png"), dpi=300)


def extract_acc_stats(all_results):
    all_test_accs = {config: {} for config in all_results.keys()}
    acc_per_config = {config: {} for config in all_results.keys()}
    for config, data in all_results.items():
        clusters = set(x["cluster_assigned"] for x in next(iter(data.values())))
        per_seed_acc = {cluster: [] for cluster in clusters}
        per_seed_acc_no_clust = []
        for seed_data in data.values():
            # final_iter = max(int(k) for k in seed_data[0]["test_acc"].keys())
            iters = [int(k) for k in seed_data[0]["test_acc"].keys()]
            per_clusters_acc = {c: [] for c in clusters}
            for x in seed_data:
                per_clusters_acc[x["cluster_assigned"]].append(x["test_acc"].values())

            mean = [np.mean(x) for x in zip(*[x["test_acc"].values() for x in seed_data])]
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


def plot_acc_per_exp(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        if data:
            iters, values, std = data
            axs[i].plot(iters, values, label=f"{exp}", color=COLORS[exp])
            axs[i].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
        axs[i].set_title(f"Accuracy for {CONFIGS[config]}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Accuracy")
        if data:
            axs[i].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_config.png"), dpi=300)


def plot_acc_per_cluster_per_exp(folder_path, all_test_accs, exp, axs):
    for i, (config, data) in enumerate(all_test_accs.items()):
        for clust, cluster_data in data.items():
            iters, values, std = cluster_data
            axs[i].plot(iters, values, label=f"{exp}: cluster {clust}", color=COLORS[exp], linestyle=LINE_STYLES[clust])
            axs[i].fill_between(
                iters, np.array(values) - np.array(std), np.array(values) + np.array(std), alpha=0.2, color=COLORS[exp]
            )
            # axs[i].errorbar(iters, values, std, label=f"Cluster {cluster}")
        axs[i].set_title(f"Accuracy for {CONFIGS[config]}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Accuracy")
        if data:
            axs[i].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_clust_per_config.png"), dpi=300)


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
    plt.savefig(os.path.join(folder_path, "settling_time_per_config.png"), dpi=300)


def compute_demo_parity(per_clust_rates, per_class_rates):
    tot_demo_par = 0
    clusters = list(per_clust_rates.keys())
    classes = list(range(len(per_class_rates[clusters[0]]["TP"])))
    print(classes)
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

        eq_op = abs(rec_0 - rec_1)
        tot_equ_oppo += eq_op
    return eq_op / len(classes)


def compute_equalized_odds(per_clust_rates, per_class_rates):
    tot_equ_odds = 0
    clusters = list(per_clust_rates.keys())
    classes = list(range(len(per_class_rates[clusters[0]]["TP"])))
    for class_ in classes:
        fpr_0 = per_class_rates[clusters[0]]["FP"][class_] / (
            per_class_rates[clusters[0]]["TN"][class_] + per_class_rates[clusters[0]]["FP"][class_]
        )
        fpr_1 = per_class_rates[clusters[1]]["FP"][class_] / (
            per_class_rates[clusters[1]]["TN"][class_] + per_class_rates[clusters[1]]["FP"][class_]
        )

        temp = abs(fpr_0 - fpr_1)
        eq_op = compute_equ_oppo(per_clust_rates, per_class_rates)
        tot_equ_odds += temp + eq_op
    # verif
    return tot_equ_odds / len(classes)


METRIC_FUNCS = {"demo_parity": compute_demo_parity, "equ_oppo": compute_equ_oppo, "eq_odds": compute_equalized_odds}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # The args are:
        # 1: the folder with the data
        plot_results(sys.argv[1])

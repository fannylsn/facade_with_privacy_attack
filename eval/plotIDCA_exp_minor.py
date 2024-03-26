import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/eval/")
from plotIDCA import (
    plot_cluster_model_evolution,
    plot_cluster_variation,
    plot_final_cluster_model_attribution,
)

CONFIGS = {
    "config1": "16:16",
    "config2": "16:8",
    "config3": "16:4",
    "config4": "16:2",
}
CONFIGS_COMP = {
    "IDCA0": "cluster 0 with IDCA",
    "IDCA1": "cluster 1 with IDCA",
    "DPSGD0": "cluster 0 with DPSGD",
    "DPSGD1": "cluster 1 with DPSGD",
}


def plot_results(folder_path, data_machine="machine0", data_node=0):
    print("Reading the folder: ", folder_path)

    all_results, is_IDCA = get_data_from_exp(folder_path)

    # plotting

    plt.figure(1)
    plot_acc_per_cluster_per_exp(folder_path, all_results)
    plt.close()
    if is_IDCA:
        plt.figure(2)
        plot_settling_time(folder_path, all_results)

    for i, (config_folder, data) in enumerate(all_results.items()):
        for j, (seed_folder, seed_data) in enumerate(data.items()):
            exp_path = os.path.join(folder_path, config_folder, seed_folder)
            if is_IDCA:
                plot_cluster_model_evolution(exp_path, seed_data)
                plot_cluster_variation(exp_path, seed_data)
                plot_final_cluster_model_attribution(exp_path, seed_data)


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
            for f in files:
                filepath = os.path.join(mf_path, f)
                with open(filepath, "r") as inf:
                    results.append(json.load(inf))
                    is_IDCA = "test_best_model_idx" in results[-1]
            config_results[seed_folder] = results
        all_results[config_folder] = config_results
    return all_results, is_IDCA


def plot_acc_per_cluster_per_exp(folder_path, all_results):
    all_test_accs = {config: {} for config in all_results.keys()}
    # TODO add loss, val, train...
    for config, data in all_results.items():
        clusters = set(x["cluster_assigned"] for x in next(iter(data.values())))
        final_iter = max(
            int(k) for k in next(iter(data.values()))[0]["test_acc"].keys()
        )
        per_seed_acc = {cluster: [] for cluster in clusters}
        for seed_data in data.values():
            per_clusters_acc = {c: [] for c in clusters}
            for x in seed_data:
                per_clusters_acc[x["cluster_assigned"]].append(
                    x["test_acc"][str(final_iter)]
                )
            for cluster, v in per_clusters_acc.items():
                per_seed_acc[cluster].append(sum(v) / len(v))  # mean across all nodes
        for cluster, v in per_seed_acc.items():
            all_test_accs[config][cluster] = [np.mean(v), np.std(v)]

    # python eval/plotIDCA_exp_minor.py eval/data/experiment_minority2024-03-15T13:57_test

    configs = list(all_test_accs.keys())
    classes = [0, 1]
    accuracy = [
        [all_test_accs[config][cls][0] for cls in classes] for config in configs
    ]
    std_dev = [[all_test_accs[config][cls][1] for cls in classes] for config in configs]

    _, _ = plt.subplots()
    bar_width = 0.35
    opacity = 0.8
    index = np.arange(len(configs))

    for cls in classes:
        plt.bar(
            index + cls * bar_width,
            [acc[cls] for acc in accuracy],
            bar_width,
            alpha=opacity,
            label=f"Cluster {cls}",
            yerr=[std[cls] for std in std_dev],
        )

    plt.xlabel("Ratio of majority (cluster 0) to minority (cluster 1)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by experiment and cluster")
    plt.xticks(index + bar_width / 2, map(lambda x: CONFIGS[x], configs))
    plt.legend()

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
            settling_iter = idx[np.max(np.nonzero(total_variations)) + 1]

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


def plot_results_all(out_folder, folder_path_IDCA, folder_path_DPSGDnIID):
    all_results_IDCA, _ = get_data_from_exp(folder_path_IDCA)
    all_results_DPSGD, _ = get_data_from_exp(folder_path_DPSGDnIID)

    all_test_accs = {config: {} for config in all_results_IDCA.keys()}
    # TODO add loss, val, train...
    for (config, data_IDCA), (_, data_DPSGD) in zip(
        all_results_IDCA.items(), all_results_DPSGD.items()
    ):
        clusters = set(x["cluster_assigned"] for x in next(iter(data_IDCA.values())))
        final_iter = max(
            int(k) for k in next(iter(data_IDCA.values()))[0]["test_acc"].keys()
        )
        per_seed_acc = {"IDCA0": [], "IDCA1": [], "DPSGD0": [], "DPSGD1": []}
        for node_type, data in {"IDCA": data_IDCA, "DPSGD": data_DPSGD}.items():
            for seed_data in data.values():
                name = [node_type + str(c) for c in clusters]
                per_clusters_acc = {c: [] for c in name}
                for x in seed_data:
                    per_clusters_acc[node_type + str(x["cluster_assigned"])].append(
                        x["test_acc"][str(final_iter)]
                    )
                for cluster, v in per_clusters_acc.items():
                    per_seed_acc[cluster].append(
                        sum(v) / len(v)
                    )  # mean across all nodes
            for cluster, v in per_seed_acc.items():
                all_test_accs[config][cluster] = [np.mean(v), np.std(v)]

    # python eval/plotIDCA_exp_minor.py eval/data/experiment_minority2024-03-15T13:57_test

    configs = list(all_test_accs.keys())
    classes = list(all_test_accs[configs[0]].keys())
    accuracy = [
        [all_test_accs[config][cls][0] for cls in classes] for config in configs
    ]
    std_dev = [[all_test_accs[config][cls][1] for cls in classes] for config in configs]

    _, _ = plt.subplots()
    bar_width = 0.2
    opacity = 0.8
    index = np.arange(len(configs))
    colors = ["darkgreen", "limegreen", "darkblue", "cornflowerblue"]

    for i, (cls, color) in enumerate(
        zip(map(lambda x: CONFIGS_COMP[x], classes), colors)
    ):
        plt.bar(
            index + i * bar_width,
            [acc[i] for acc in accuracy],
            bar_width,
            alpha=opacity,
            label=f"Cluster {cls}",
            yerr=[std[i] for std in std_dev],
            color=color,
        )

    plt.xlabel("Ratio of majority (cluster 0) to minority (cluster 1)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by experiment and cluster")
    plt.xticks(index + bar_width * 1.5, map(lambda x: CONFIGS[x], configs))
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "acc_per_clust_per_config.png"), dpi=300)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # The args are:
        # 1: the folder with the data
        plot_results(sys.argv[1])
    elif len(sys.argv) == 4:
        # The args are:
        # 1: the folder to put the plots
        # 2: the folder with the data of IDCA
        # 3: the folder with the data of DPSGDnIID
        plot_results_all(sys.argv[1], sys.argv[2], sys.argv[3])

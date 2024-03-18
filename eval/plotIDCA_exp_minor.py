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


def plot_results(folder_path, data_machine="machine0", data_node=0):
    print("Reading the folder: ", folder_path)
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
            config_results[seed_folder] = results
        all_results[config_folder] = config_results

    # plotting

    plt.figure(1)
    plot_acc_per_cluster_per_exp(folder_path, all_results)

    for i, (config_folder, data) in enumerate(all_results.items()):
        for j, (seed_folder, seed_data) in enumerate(data.items()):
            exp_path = os.path.join(folder_path, config_folder, seed_folder)
            plot_cluster_model_evolution(exp_path, seed_data)
            plot_cluster_variation(exp_path, seed_data)
            plot_final_cluster_model_attribution(exp_path, seed_data)


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
    plt.title("Accuracy by Configuration and Class")
    plt.xticks(index + bar_width / 2, map(lambda x: CONFIGS[x], configs))
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "acc_per_clust_per_config.png"), dpi=300)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: the folder with the data
    plot_results(sys.argv[1])
    # plot_parameters(sys.argv[1])

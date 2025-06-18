import json
import math
import os
import sys
import torch
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

def get_results(folder_path: str) -> list:
    """
    Retrieve results from JSON files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing result files.

    Returns:
        list: A list of dictionaries containing the results.
    """
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

    return results


def get_attack_data(folder_path: str) -> dict:
    """
    Retrieve attack data from PyTorch files in the specified folder. It organizes the data into 
    a nested dictionary structure where the first level keys are victim client identifiers, 
    the second level keys are iteration numbers, and the values are lists of attack data entries.

    Args:
        folder_path (str): Path to the folder containing attack data files.

    Returns:
        dict: A nested dictionary containing the attack data. The structure is as follows:
            {
                "victim_client_1": {
                    "iteration_1": [attack_data_entry_1, attack_data_entry_2, ...],
                    "iteration_2": [attack_data_entry_1, attack_data_entry_2, ...],
                    ...
                },
                "victim_client_2": {
                    "iteration_1": [attack_data_entry_1, attack_data_entry_2, ...],
                    "iteration_2": [attack_data_entry_1, attack_data_entry_2, ...],
                    ...
                },
                ...
            }
            where each `attack_data_entry` is a list of loss values collected by an attacker.
    """
    loss_attack_dict = {}
    machine_folders = os.listdir(folder_path)

    for machine_folder in machine_folders:
        mf_path = os.path.join(folder_path, machine_folder)
        if not os.path.isdir(mf_path):
            continue

        files = [f for f in os.listdir(mf_path) if f.endswith("_attacker.pth")]

        for f in files:
            filepath = os.path.join(mf_path, f)
            try:
                attacker_file = torch.load(filepath, weights_only=True)

                for victim_client in attacker_file['loss_vals'].keys():
                    if victim_client not in loss_attack_dict:
                        loss_attack_dict[victim_client] = {}

                    for iteration in attacker_file['loss_vals'][victim_client].keys():
                        if iteration not in loss_attack_dict[victim_client]:
                            loss_attack_dict[victim_client][iteration] = []

                        loss_attack_dict[victim_client][iteration].extend(attacker_file['loss_vals'][victim_client][iteration])

            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

    return loss_attack_dict


def plot_loss_distribution(data: dict, cluster_id: int, node: int, folder_path: str) -> None:
    """
    Plot the loss distributions for a given node in a cluster.

    Args:
        data (dict): Dictionary containing loss data.
        cluster_id (int): ID of the cluster.
        node (int): ID of the node.
        folder_path (str): Path to the folder where the plot will be saved.
    """
    in_all = []
    out_all = []

    for value in data.values():
        for j in range(len(value)):
            in_all.append(value[j]['in'].detach().cpu().numpy())
            out_all.append(value[j]['out'].detach().cpu().numpy())

    in_all = np.concatenate(in_all)
    out_all = np.concatenate(out_all)

    all_vals = np.concatenate([in_all, out_all])
    mean = np.mean(all_vals)
    std = np.std(all_vals)

    in_all = (in_all - mean) / std
    out_all = (out_all - mean) / std

    plt.figure(figsize=(8, 5))
    sns.ecdfplot(in_all, label='member')
    sns.ecdfplot(out_all, label='non-member')
    plt.title(f'Loss Distributions for Node {node} in Cluster {cluster_id}')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{folder_path}/loss_distribution_c{cluster_id}_n{node}.png", dpi=300)


def get_roc_auc(a: dict) -> tuple:
    """
    Calculate the ROC AUC for given data.

    Args:
        a (dict): Dictionary containing 'in' and 'out' data.

    Returns:
        tuple: False positive rates, true positive rates, thresholds, and ROC AUC score.
    """
    number_true = a["in"].numel()
    number_false = a["out"].numel()

    y_true_balanced = torch.zeros((number_true + number_false,), dtype=torch.int32)
    y_true_balanced[:number_true] = 1

    y_pred_balanced = torch.zeros((number_true + number_false,), dtype=torch.float32)
    y_pred_balanced[:number_true] = a["in"]
    y_pred_balanced[number_true:] = a["out"]

    y_pred = y_pred_balanced.numpy()
    y_true = y_true_balanced.numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def get_auc_means_iterations(attack_dict: dict, iterations_to_attack: list) -> tuple:
    """
    Calculate the mean and standard deviation of AUC values for given iterations.

    Args:
        attack_dict (dict): Dictionary containing attack data.
        iterations_to_attack (list): List of iteration keys to calculate AUC for.

    Returns:
        tuple: Mean AUC values, standard deviation of AUC values, and counts.
    """
    auc_means_iterations = []
    auc_stdev_iterations = []
    counts = []

    for iteration in iterations_to_attack:
        aucs = []
        for victim_client in attack_dict:
            if iteration in attack_dict[victim_client].keys():
                for attack_by_each_attacker in attack_dict[victim_client][iteration]:
                    _, _, _, roc_auc = get_roc_auc(attack_by_each_attacker)
                    aucs.append(roc_auc)

        aucs = torch.tensor(aucs)
        auc_means_iterations.append(torch.mean(aucs))
        auc_stdev_iterations.append(torch.std(aucs) / math.sqrt(aucs.numel()) if aucs.numel() > 1 else torch.tensor(0.0))
        counts.append(aucs.numel())

    return torch.tensor(auc_means_iterations), torch.tensor(auc_stdev_iterations), torch.tensor(counts)


def plot_final_iteration_ROC(folder_path: str, attack_dict: dict, results: list) -> None:
    """
    Plot the ROC curve for the final iteration of an attack on each client.

    Args:
        folder_path (str): Path to the folder where the plot will be saved.
        attack_dict (dict): Dictionary containing attack data.
        results (list): List of result dictionaries.
    """
    clients = attack_dict.keys()

    for client in clients:
        plt.clf()
        plt.figure(figsize=(12, 8))

        iterations = sorted(attack_dict[client].keys())
        final_iteration = iterations[-1] if iterations else None

        attack_count = len(attack_dict[client][final_iteration])
        print(f"Client {client} has been attacked {attack_count} times in the final iteration {final_iteration}")

        if final_iteration is not None:
            for attack in attack_dict[client][final_iteration]:
                fpr, tpr, _, roc_auc = get_roc_auc(attack)
                plt.plot(fpr, tpr, lw=1, label=f"Final Iteration {final_iteration} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'Receiver Operating Characteristic (ROC) for Client {client} at Final Iteration', fontsize=16)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(folder_path, f"ROC_curve_client_{client}_final_iteration.png"), dpi=300)
        plt.close()


def plot_per_cluster_ROC(folder_path: str, attack_dict: dict, results: list) -> None:
    """
    Plot the ROC curve for an attack on each cluster.

    Args:
        folder_path (str): Path to the folder where the plot will be saved.
        attack_dict (dict): Dictionary containing attack data.
        results (list): List of result dictionaries.
    """
    cluster_assign = [x["cluster_assigned"] for x in results]
    clusters_idxs = set(cluster_assign)
    nb_nodes = len(results)

    iterations_to_attack = sorted(set(iteration for client_attacks in attack_dict.values() for iteration in client_attacks.keys()))

    plt.figure(figsize=(12, 8))
    fpr_axis = np.linspace(0, 1, 100)

    for cluster in clusters_idxs:
        cluster_clients = [idx for idx, clust in enumerate(cluster_assign) if clust == cluster]
        minority_cluster = len(cluster_clients) < (nb_nodes / len(clusters_idxs))

        attack_dict_cluster = {client: attack_dict[client] for client in cluster_clients if client in attack_dict}

        tprs = []
        aucs = []

        for iteration in iterations_to_attack:
            for client in attack_dict_cluster.keys():
                if iteration in attack_dict_cluster[client]:
                    for attack in attack_dict_cluster[client][iteration]:
                        fpr, tpr, _, roc_auc = get_roc_auc(attack)
                        aucs.append(roc_auc)
                        tpr_interp = np.interp(fpr_axis, fpr, tpr)
                        tprs.append(tpr_interp)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)

        label = f"Cluster {cluster} (Minority)" if minority_cluster else f"Cluster {cluster}"

        plt.plot(fpr_axis, mean_tpr, label=f"{label} (AUC = {np.mean(mean_auc):.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) for attack on each cluster', fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(folder_path, f"ROC_curve_per_cluster.png"), dpi=300)


def plot_per_cluster_AUC(folder_path: str, attack_dict: dict, results: list) -> None:
    """
    Plot the mean AUC of the LOSS-MIA per cluster.

    Args:
        folder_path (str): Path to the folder where the plot will be saved.
        attack_dict (dict): Dictionary containing attack data.
        results (list): List of result dictionaries.
    """
    print("Processing MIA Attack")
    print("---------------------")

    cluster_assign = [x["cluster_assigned"] for x in results]
    clusters_idxs = set(cluster_assign)
    nb_nodes = len(results)
    minority_exp = False

    iterations_to_attack = sorted(set(iteration for client_attacks in attack_dict.values() for iteration in client_attacks.keys()))
    plt.figure(figsize=(12, 8))

    for cluster in clusters_idxs:
        print(f"Processing cluster {cluster}")
        cluster_clients = [idx for idx, clust in enumerate(cluster_assign) if clust == cluster]
        print(f"Cluster {cluster} has {len(cluster_clients)} clients")

        if len(cluster_clients) < (nb_nodes / len(clusters_idxs)):
            minority_cluster = True
            minority_exp = True
            len_minority = len(cluster_clients)
        else:
            minority_cluster = False

        attack_dict_cluster = {client: attack_dict[client] for client in cluster_clients if client in attack_dict}

        if not attack_dict_cluster:
            print(f"No attack data for cluster {cluster}, skipping...")
            continue

        auc_means_iterations, auc_stdev_iterations, counts_iterations = get_auc_means_iterations(attack_dict_cluster, iterations_to_attack)

        df = pd.DataFrame({
            "Iterations": iterations_to_attack,
            "mean": auc_means_iterations,
            "std": auc_stdev_iterations.numpy(),
            "nr_nodes": counts_iterations.numpy()
        })

        df.to_csv(f"{folder_path}_iterations_MIA_{cluster}.csv", index=False)

        label = f"Cluster {cluster} (Minority)" if minority_cluster else f"Cluster {cluster}"
        plt.plot(iterations_to_attack, auc_means_iterations, label=label)
        plt.fill_between(
            iterations_to_attack,
            auc_means_iterations - auc_stdev_iterations.numpy(),
            auc_means_iterations + auc_stdev_iterations.numpy(),
            alpha=0.1,
            lw=2
        )

    plt.axhline(y=0.5, color="navy", lw=1, linestyle="--")
    ratio = f"\nRatio: {nb_nodes - len_minority}:{len_minority}" if minority_exp else ""
    plt.title(f"Mean AUC of LOSS-MIA per cluster {ratio}", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(iterations_to_attack, fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right")
    plt.ylim(0.4, 1)
    plt.savefig(os.path.join(folder_path, f"AUC_iterations_per_cluster.png"), dpi=300)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    folder_path = Path(os.path.abspath(sys.argv[1]))
    attack_data = get_attack_data(folder_path)
    results = get_results(folder_path)
    cluster_assign = [x["cluster_assigned"] for x in results]

    for i in range(2):
        cluster_clients = [idx for idx, cluster in enumerate(cluster_assign) if cluster == i]
        cluster_victims_data = {client: attack_data[client] for client in cluster_clients if client in attack_data}
        for client, data in cluster_victims_data.items():
            plot_loss_distribution(data, i, client, folder_path)

    iterations_to_attack = sorted(set(iteration for client_attacks in attack_data.values() for iteration in client_attacks.keys()))

    plot_per_cluster_AUC(folder_path, attack_data, results)
    plot_per_cluster_ROC(folder_path, attack_data, results)
    plot_final_iteration_ROC(folder_path, attack_data, results)
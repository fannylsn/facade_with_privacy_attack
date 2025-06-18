import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
sns.set_palette("bright")
sns.set_palette([sns.color_palette()[3], sns.color_palette()[0], sns.color_palette()[2]])

def plot_side_by_side(dfs, titles, hue_type, save_as, y_lim=(0.44, 0.9), y_text=0.49):
    """
    Plot 3 plots side-by-side with a shared legend.

    Parameters:
        dfs (list): List of 3 DataFrames to plot.
        titles (list): List of 3 titles corresponding to each plot.
        hue_type (str): Type of hue to use for the plot. Can be 'cluster_ratio' or 'head_layers'.
        y_lim (tuple): Tuple specifying the y-axis limits. Default is (0.4, 0.9).
        y_text (float): Y-coordinate for the 'Random Guessing' text. Default is 0.49.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    all_handles = []
    all_labels = []

    for ax, df, title in zip(axes, dfs, titles):
        # Plot
        sns.lineplot(
            data=df,
            x='Iterations',
            y='mean',
            hue=hue_type,
            style='cluster_id',
            dashes={'0': '', '1 (minority except for 8-8 ratio)': (1, 1), '1': (1, 1)},
            linewidth=1.7,
            ax=ax
        )
        ax.set_title(title, fontsize='18')
        ax.set_xlabel('Training Rounds', fontsize='15')
        ax.set_ylabel('Mean AUC', fontsize='15')
        ax.set_ylim(y_lim)
        ax.set_xlim(0, 1000)
        ax.grid(True)
        ax.axhline(y=0.5, color='gray', linestyle='--')
        ax.text(400, y_text, 'Random Guessing', color='gray', fontsize=10, verticalalignment='center')

        # Collect handles/labels for legend
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
        ax.legend_.remove()  # Remove individual legend

    # Remove duplicate labels (to prevent duplicates in combined legend)
    seen = set()
    unique_handles_labels = [(h, l) for h, l in zip(all_handles, all_labels) if not (l in seen or seen.add(l))]

    # Shared legend
    fig.legend(
        *zip(*unique_handles_labels),
        loc='upper center',
        ncol=7,
        fontsize='15',
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space at top for legend
    plt.savefig(os.path.join(os.getcwd(), 'plots', save_as), bbox_inches='tight', dpi=300)

def map_cluster_id(row):
    return '0' if row['cluster_id'] == 0 else '1 (minority except for 8-8 ratio)'

def load_and_preprocess_data(path_to_csv):
    # iterate over all csv files in the path_to_csv folder
    ratios = []
    for f in os.listdir(path_to_csv):
        if f.endswith('.csv'):
            print(f"Found CSV file: {f}")
            ratio = f.split('_')[0].replace('ratio', '')
            df = pd.read_csv(os.path.join(path_to_csv, f))
            df['cluster_ratio'] = ratio
            ratios.append(df)

    # Concatenate all dataframes into one
    merged_df = pd.concat(ratios, ignore_index=True)

    # Some preprocessing steps
    merged_df['head_layers'] = [int(l) if (isinstance(l, str) and (l=='1' or l=='2')) else l for l in merged_df['head_layers']]
    merged_df['cluster_id'] = merged_df.apply(map_cluster_id, axis=1)

    return merged_df


if __name__ == "__main__":
    assert len(sys.argv) == 2

    path_to_csv = sys.argv[1]
    data = load_and_preprocess_data(path_to_csv)
    # drop cluster_ratio 10:6 as it is not used in the end
    data = data[data['cluster_ratio'] != '10:6']

    # Prepare DataFrames for plotting
    titles = [
        'Non-member dataset\nUNION(cluster 0 testset, cluster 1 testset)',
        'Non-member dataset\ncluster 0 testset',
        'Non-member dataset\ncluster 1 (minority) testset'
    ]

    ratio_comparison_df = [
        data[(data['head_layers'] == 1) & (data['test_set'] == 'union')],
        data[(data['head_layers'] == 1) & (data['test_set'] == 'test_cluster_0')],
        data[(data['head_layers'] == 1) & (data['test_set'] == 'test_cluster_1')]
    ]
    plot_side_by_side(ratio_comparison_df, titles, hue_type='cluster_ratio', 
                      save_as='cluster_ratio_comparison.png', y_lim=(0.44, 0.81))


    head_layers_comparison_df1 = [
        data[(data['cluster_ratio'] == '8-8') & (data['test_set'] == 'union')],
        data[(data['cluster_ratio'] == '8-8') & (data['test_set'] == 'test_cluster_0')],
        data[(data['cluster_ratio'] == '8-8') & (data['test_set'] == 'test_cluster_1')]
    ]
    plot_side_by_side(head_layers_comparison_df1, titles, hue_type='head_layers', save_as='head_layers_comparison_8-8.png')

    head_layers_comparison_df2 = [
        data[(data['cluster_ratio'] == '12-4') & (data['test_set'] == 'union')],
        data[(data['cluster_ratio'] == '12-4') & (data['test_set'] == 'test_cluster_0')],
        data[(data['cluster_ratio'] == '12-4') & (data['test_set'] == 'test_cluster_1')]
    ]
    plot_side_by_side(head_layers_comparison_df2, titles, hue_type='head_layers', save_as='head_layers_comparison_12-4.png')

    head_layers_comparison_df3 = [
        data[(data['cluster_ratio'] == '14-2') & (data['test_set'] == 'union')],
        data[(data['cluster_ratio'] == '14-2') & (data['test_set'] == 'test_cluster_0')],
        data[(data['cluster_ratio'] == '14-2') & (data['test_set'] == 'test_cluster_1')]
    ]
    plot_side_by_side(head_layers_comparison_df3, titles, hue_type='head_layers', save_as='head_layers_comparison_14-2.png')

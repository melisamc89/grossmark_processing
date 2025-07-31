

import pandas as pd
import numpy as np
import pickle as pkl

from src.config import *
from src.channel_information import *

from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate
from quantities import ms
import neo
import umap
from src.general_utils import *
from scipy.ndimage import gaussian_filter1d
import random
import matplotlib.pyplot as plt
from src.structure_index import *

def spikes_to_rates(spikes, kernel_width=10):
    gk = GaussianKernel(kernel_width * ms)
    rates = []
    for sp in spikes:
        sp_times = np.where(sp)[0]
        st = neo.SpikeTrain(sp_times, units="ms", t_stop=len(sp))
        r = instantaneous_rate(st, kernel=gk, sampling_period=1. * ms).magnitude
        rates.append(r.T)

    rates = np.vstack(rates)

    return rates.T

params1 = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 0,
    'verbose': False
}


params2 = {
    'n_bins': 3,
    'discrete_label': True,
    'continuity_kernel': None,
    'n_neighbors': 100,
    'num_shuffles': 0,
    'verbose': False
}

si_neigh = 100
si_beh_params = {}
for beh in ['pos', 'speed', 'trial_id_mat','time','(pos,dir)']:
    si_beh_params[beh] = copy.deepcopy(params1)
for beh in ['dir']:
    si_beh_params[beh] = copy.deepcopy(params2)


neural_data_dir = files_directory

from src.config import *
# Setup
cluster_directory = os.path.join(base_directory, 'clusters')

files_names = ['Achilles_mi_transferred_cluster_3_dict.pkl',
               'Calb_mi_transferred_cluster_3_dict.pkl',
               'Thy_mi_transferred_cluster_3_dict.pkl']
mouse_names = ['Achilles', 'Calb', 'Thy']


files_names = ['Calb_mi_transferred_cluster_3_dict.pkl',
               'Thy_mi_transferred_cluster_3_dict.pkl']
mouse_names = ['Calb', 'Thy']


behavior_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']

# Initialize storage
mouse_name_list, area_list, probe_list, typeID_list = [], [], [], []
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}
import pickle
# Load data
speed_limit = {'Achilles': 0.05, 'Calb': 6, 'Thy': 6}
for fname, mouse in zip(files_names, mouse_names):
    filepath = os.path.join(cluster_directory, fname)
    with open(filepath, 'rb') as f:
        mi_dict = pickle.load(f)
    si_dict = dict()
    for probe, probe_data in mi_dict.items():
        si_dict[probe] = dict()
        signal = probe_data['signal']
        clusterID = probe_data['clusterID']

        beh_variables = {
            'pos':  probe_data['behaviour']['position'],
            '(pos,dir)': probe_data['behaviour']['(pos,dir)'],
            'speed':probe_data['behaviour']['speed'],
            'trial_id_mat': probe_data['behaviour']['trial_id'],
            'dir':probe_data['behaviour']['mov_dir'],
            'time':probe_data['behaviour']['time'],
        }
        valid_index = np.where(probe_data['behaviour']['speed'] > speed_limit[mouse])
        clusters_names = np.unique(clusterID)
        for beh_name, beh_val in beh_variables.items():
            si_dict[probe][beh_name] = dict()
            if 'trial_id_mat' in beh_name:
                si_beh_params[beh_name]['min_label'] = np.min(beh_val)
                si_beh_params[beh_name]['max_label'] = np.max(beh_val)
            # si_beh_params[beh_name]['n_neighbors'] = int(signal.shape[0]* si_beh_params[beh_name]['perc_neigh']/100)
            si_beh_params[beh_name]['n_neighbors'] = si_neigh
            si_dict[beh_name] = dict()
            si, process_info, overlap_mat, shuff_si = compute_structure_index(signal[valid_index],
                                                                              beh_val[valid_index],
                                                                              **si_beh_params[
                                                                                  beh_name])

            si_dict[probe][beh_name][str(-1)] = {
                'si': copy.deepcopy(si),
                'si_shuff': copy.deepcopy(shuff_si),
                # 'si_umap': cop copy.deepcopy(si),y.deepcopy(si_umap),
                'si_umap': 0,
                'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                'valid_idx': valid_index,
                'signal': signal.copy()
            }
            for cluster_idx in clusters_names:
                if cluster_idx == 10: continue;
                cluster_signal = signal[:, clusterID == cluster_idx]

                si, process_info, overlap_mat, si_shuff = compute_structure_index(cluster_signal[valid_index],
                                                                                  beh_val[valid_index],
                                                                                  **si_beh_params[beh_name])

                si_dict[probe][beh_name][str(cluster_idx)] = {
                    'si': copy.deepcopy(si),
                    'si_shuff': copy.deepcopy(shuff_si),
                    # 'si_umap': copy.deepcopy(si_umap),
                    'si_umap': 0,
                    'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                    'valid_idx': valid_index,
                    'signal': cluster_signal.copy()
                }

                print(f" {beh_name}_{cluster_idx}={si:.4f} |", end='', sep='', flush='True')
                print()

    data_output_directory = '/home/melma31/Documents/time_project/SI_clusters'
    os.makedirs(os.path.dirname(data_output_directory ), exist_ok=True)
    # Define the filename where the dictionary will be stored
    output_filename = mouse + '_si_clusters_100nei.pkl'
    # Open the file for writing in binary mode and dump the dictionary
    with open(os.path.join(data_output_directory, output_filename), 'wb') as file:
        pkl.dump(si_dict, file)

##################################################################################################

# CREATE DF WITH si AND PLOT

##################################################################################################
import os
import pickle as pkl
import pandas as pd
# Directory containing your SI pickle files
data_output_directory = '/home/melma31/Documents/time_project/SI_clusters'
all_data = []
# List your mice (you can also use glob if needed)
mice = ['Achilles','Calb', 'Thy']
for mouse in mice:
    filename = os.path.join(data_output_directory, f'{mouse}_si_clusters_100nei.pkl')
    with open(filename, 'rb') as f:
        si_dict = pkl.load(f)
    for probe in si_dict:
        for beh_name in si_dict[probe]:
            for cluster_id_str, data in si_dict[probe][beh_name].items():
                all_data.append({
                    'mouseId': mouse,
                    'probe': probe,
                    'behavioural_name': beh_name,
                    'cluster_ID': int(cluster_id_str),
                    'SI_value': data['si']
                })
# Create dataframe
si_df = pd.DataFrame(all_data)
# Save final dataframe
output_df_path = os.path.join(data_output_directory, 'all_mice_SI_dataframe.pkl')
si_df.to_pickle(output_df_path)
print(f"Saved combined SI dataframe to {output_df_path}")

behavior_keys = [
    'pos',
    'dir',
    '(pos,dir)',
    'speed',
    'time',
    'trial_id_mat']


from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
palette = {'SUPERFICIAL': '#9900ff', 'DEEP': '#cc9900'}


cluster_palette = {
    '10': '#bbbcc0ff',
     '0': '#bce784ff',        # green-ish
     '1':  '#66cef4ff',        #  blue-ish
     '2':  '#ec8ef8ff',       # red-ish
}

label_colors = {
    10: '#bbbcc0ff',
    0: '#bce784ff',        # green-ish
     1:  '#66cef4ff',        #  blue-ish
     2:  '#ec8ef8ff',       # red-ish
}


# Cluster pairs to compare (including -1)
cluster_ids = [-1, 0, 1, 2]
cluster_pairs = list(itertools.combinations(cluster_ids, 2))

# Manual bar coloring approach using matplotlib for robustness
fig, axes = plt.subplots(1, 6, figsize=(20, 5), sharey=True)
axes = axes.flatten()

for i, beh in enumerate(behavior_keys):
    ax = axes[i]
    subset = si_df[si_df['behavioural_name'] == beh].copy()
    subset['cluster_ID'] = subset['cluster_ID'].astype(int)

    present_clusters = sorted(subset['cluster_ID'].unique())

    # Compute means and stds
    means = subset.groupby('cluster_ID')['SI_value'].mean()
    stds = subset.groupby('cluster_ID')['SI_value'].std()
    counts = subset.groupby('cluster_ID')['SI_value'].count()

    # Plot bars manually
    for j, cluster_id in enumerate(present_clusters):
        mean = means[cluster_id]
        std = stds[cluster_id]
        count = counts[cluster_id]
        color = cluster_palette.get(str(cluster_id), 'gray')
        ax.bar(j, mean, yerr=std / count**0.5, color= color, edgecolor='black', capsize=5)

    # Overlay individual dots
    for j, cluster_id in enumerate(present_clusters):
        values = subset[subset['cluster_ID'] == cluster_id]['SI_value']
        ax.scatter([j]*len(values), values, color='black', s=20, alpha=0.6)

    ax.set_xticks(range(len(present_clusters)))
    ax.set_xticklabels(present_clusters)
    ax.set_title(beh, fontsize=12)
    ax.set_xlabel('Cluster ID')
    if i == 0:
        ax.set_ylabel('SI Value')

    # Statistical annotations
    y_max = subset['SI_value'].max()
    height_step = 0.05 * y_max
    base_height = y_max + 0.05 * y_max

    from scipy.stats import ttest_ind
    # Statistical annotations
    y_max = subset['SI_value'].max()
    height_step = 0.05 * y_max
    base_height = y_max + 0.05 * y_max
    for k, (c1, c2) in enumerate(cluster_pairs):
        if c1 in present_clusters and c2 in present_clusters:
            data1 = subset[subset['cluster_ID'] == c1]['SI_value']
            data2 = subset[subset['cluster_ID'] == c2]['SI_value']
            if len(data1) > 0 and len(data2) > 0:
                stat, p = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = None

                if sig:
                    x1 = present_clusters.index(c1)
                    x2 = present_clusters.index(c2)
                    y = base_height + k * height_step
                    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.5, c='k')
                    ax.text((x1 + x2) / 2, y + 0.01, sig, ha='center', va='bottom', color='k', fontsize=10)
plt.tight_layout()
plt.suptitle('SI per Behavioral Variable and Cluster with Significance', fontsize=16, y=1.05)
output_dir = os.path.join(figures_directory, 'SI')
os.makedirs(output_dir, exist_ok=True)
fig_path_base = os.path.join(output_dir, 'SI_per_behavior_cluster')
plt.savefig(f"{fig_path_base}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{fig_path_base}.svg", bbox_inches='tight')
plt.show()



cluster_palette_numbers = {
    -1: '#bbbcc0ff',
    0: '#bce784ff',        # green-ish
     1:  '#66cef4ff',        #  blue-ish
     2:  '#ec8ef8ff',       # red-ish
}


# Ensure cluster_ID is integer
si_df['cluster_ID'] = si_df['cluster_ID'].astype(int)

# Pivot the dataframe to get SI values for 'pos' and 'time' per cluster
pivot_df = si_df[si_df['behavioural_name'].isin(['pos', 'time'])]
pivot_df = pivot_df.pivot_table(index=['mouseId', 'probe', 'cluster_ID'],
                                columns='behavioural_name',
                                values='SI_value').reset_index()

# Drop rows with any missing SI values
pivot_df = pivot_df.dropna(subset=['pos', 'time'])
colors = pivot_df['cluster_ID'].map(cluster_palette_numbers)

# Scatter plot
plt.figure(figsize=(7, 6))
plt.scatter(pivot_df['time'], pivot_df['pos'], c=colors, alpha=0.7, s=60, edgecolors='k')
plt.xlabel('SI (Time)')
plt.ylabel('SI (Pos)')
plt.title('SI Position vs Time colored by Cluster ID')

# Add legend
for cid, color in cluster_palette.items():
    plt.scatter([], [], c=color, label=f'Cluster {cid}', s=60)
plt.legend(title='Cluster ID')

plt.grid(False)
plt.xlim([0,1])
plt.ylim([0,1])
plt.tight_layout()
output_dir = os.path.join(figures_directory, 'SI')
os.makedirs(output_dir, exist_ok=True)
fig_path_base = os.path.join(output_dir, 'SI_pos_vs_time_cluster')
plt.savefig(f"{fig_path_base}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{fig_path_base}.svg", bbox_inches='tight')
plt.show()




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
from sklearn.feature_selection import mutual_info_regression


import matplotlib.pyplot as plt

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

neural_data_dir = files_directory

rat_index = [0]
sessions = [[0]]
speed_lim = 0.0

for rat_index in [0,1]:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        mi_dict = {}
        for probe in ['Probe1','Probe2']:
            mi_dict[probe] = dict()
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode

            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)

            position = stimes['position']
            time_stamps = stimes['timeStamps']
            speed = abs(get_speed(position, time_stamps))
            direction = get_directions(position)
            trial_id = get_trial_id(position)
            internal_time = compute_internal_trial_time(trial_id, 40)
            time = time_stamps

            invalid_mov = np.logical_or(speed < speed_lim, speed < speed_lim)
            valid_mov = np.logical_not(invalid_mov)
            valid_index = np.arange(0, direction.shape[0])

            pos = position[valid_index]
            speed = speed[valid_index]
            mov_dir = direction[valid_index]
            time_stamps = time_stamps[valid_index]
            trial_id = trial_id[valid_index]
            inner_time = internal_time[valid_index]
            time = time[valid_index]
            pos_dir = position*mov_dir

            k = 10
            spikes_matrix = stimes['spikes_matrix']
            spikes_matrix = spikes_matrix[valid_index,:]
            signal = spikes_to_rates(spikes_matrix.T, kernel_width=k)

            layerID = stimes['LayerID']
            typeID = stimes['TypeID']

            behaviours_list = [pos, pos_dir, mov_dir, speed, time, inner_time, trial_id]
            beh_names = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
            behaviour_dict = {
                'position': behaviours_list[0],
                '(pos,dir)': behaviours_list[1],
                'mov_dir': behaviours_list[2],
                'speed': behaviours_list[3],
                'time': behaviours_list[4],
                'inner_time': behaviours_list[5],
                'trial_id': behaviours_list[6]
            }

            mi_all = []
            for beh_index, beh in enumerate(behaviours_list):
                print('MI for variable:' + beh_names[beh_index])
                mi = []
                for neui in range(signal.shape[1]):
                    neuron_mi = \
                        mutual_info_regression(signal[valid_index, neui].reshape(-1, 1), beh[valid_index], n_neighbors=50,
                                               random_state=16)[0]
                    mi.append(neuron_mi)
                # mi_stack = np.vstack([mi,mi2,mi3]).T
                mi_all.append(mi)
            # mi_final = np.hstack(mi_all)
            mi_dict[probe]['behaviour'] = behaviour_dict
            mi_dict[probe]['signal'] = signal
            #mi_dict[probe]['valid_index'] = valid_index
            mi_dict[probe]['MIR'] = mi_all
            mi_dict[probe]['area'] = layerID
            mi_dict[probe]['typeID'] = typeID

            output_filename = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] + '_mutual_information_output.pkl'
            # Open the file for writing in binary mode and dump the dictionary
            if not os.path.exists(files_directory):
                os.makedirs(files_directory)
            # Define the filename where the dic
            with open(os.path.join(files_directory, output_filename), 'wb') as file:
                pkl.dump(mi_dict, file)

#####################################################################
####################################################################
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore

# Setup
MIR_directory = os.path.join(base_directory, 'MIR')
files_names = ['Achilles_MIR.pkl', 'Calb_MIR.pkl', 'Thy_MIR.pkl']
mouse_names = ['Achilles', 'Calb', 'Thy']
behavior_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']

# Initialize storage
mouse_name_list, area_list, probe_list, typeID_list = [], [], [], []
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}

# Load data
for fname, mouse in zip(files_names, mouse_names):
    filepath = os.path.join(MIR_directory, fname)
    with open(filepath, 'rb') as f:
        mi_dict = pickle.load(f)

    for probe, probe_data in mi_dict.items():
        mi_all = np.array(probe_data['MIR'])  # shape (7, num_neurons)
        area = probe_data['area']             # list of area labels, length = num_neurons
        typeID = probe_data['typeID']         # list of typeIDs, length = num_neurons
        mi_z = zscore(mi_all, axis=1)

        num_neurons = mi_all.shape[1]

        for neuron_idx in range(num_neurons):
            mouse_name_list.append(mouse)
            probe_list.append(probe)
            area_list.append(area[neuron_idx])
            typeID_list.append(typeID[neuron_idx])

            for i, key in enumerate(behavior_keys):
                raw_mi_values[key].append(mi_all[i, neuron_idx])
                z_mi_values[f'z_{key}'].append(mi_z[i, neuron_idx])

# Create final DataFrame
mi_pd = pd.DataFrame({
    'mouse': mouse_name_list,
    'probe': probe_list,
    'area': area_list,
    'typeID': typeID_list,
    **raw_mi_values,
    **z_mi_values
})

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
# Compute total MIR per neuron
mi_pd['total_mir'] = mi_pd[['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']].sum(axis=1)
# Filter for PYR cells and total_mir >= 0.5
mi_pd_pyr = mi_pd[(mi_pd['typeID'] == 'PYR')].reset_index(drop=True)
#mi_pd_pyr = mi_pd[(mi_pd['total_mir'] >= 0)].reset_index(drop=True)
#mi_pd_pyr = mi_pd.copy()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator

# 1. Prepare long-format data
mi_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']
mi_long = mi_pd_pyr.melt(value_vars=mi_keys, var_name='Behavior', value_name='MI_value')

# 2. Kruskal-Wallis test across all groups
stat, p_kruskal = kruskal(*[mi_long[mi_long['Behavior'] == key]['MI_value'] for key in mi_keys])
print(f"Kruskal-Wallis H-statistic = {stat:.3f}, p = {p_kruskal:.5f}")

# 3. Pairwise Mann-Whitney U tests
pairs = [(a, b) for i, a in enumerate(mi_keys) for b in mi_keys[i+1:]]
p_values = []
for a, b in pairs:
    u_stat, p = mannwhitneyu(mi_long[mi_long['Behavior'] == a]['MI_value'],
                             mi_long[mi_long['Behavior'] == b]['MI_value'],
                             alternative='two-sided')
    p_values.append(p)

# 4. FDR correction
reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# 5. Plot violin + annotate
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Behavior', y='MI_value', data=mi_long, inner='quartile', palette='coolwarm')
plt.title('Raw MI Values per Behavior (PYR cells)')
plt.ylabel('Mutual Information')
plt.xlabel('Behavior')

# Annotate significant pairs
annotator = Annotator(ax, pairs, data=mi_long, x='Behavior', y='MI_value')
annotator.set_pvalues(pvals_corrected.tolist())
annotator.annotate()

plt.tight_layout()
plt.show()

area_palette = {'DEEP': '#FFD700', 'SUPERFICIAL': '#800080'}

# Create subplots
fig, axs = plt.subplots(1, 7, figsize=(24, 5), sharey=True)
for i, key in enumerate(mi_keys):
    ax = axs[i]
    df_plot = mi_pd_pyr[['area', key]].rename(columns={key: 'MI_value'}).copy()
    # Plot
    sns.violinplot(x='area', y='MI_value', data=df_plot, palette=area_palette, ax=ax, inner='quartile')
    ax.set_title(key)
    ax.set_xlabel('')
    if i == 0:
        ax.set_ylabel('Mutual Information')
    else:
        ax.set_ylabel('')
    # Stats between deep and superficial
    deep_vals = df_plot[df_plot['area'] == 'DEEP']['MI_value']
    sup_vals = df_plot[df_plot['area'] == 'SUPERFICIAL']['MI_value']
    stat, p = mannwhitneyu(deep_vals, sup_vals, alternative='two-sided')
    # Annotate
    annotator = Annotator(ax, [('DEEP', 'SUPERFICIAL')], data=df_plot, x='area', y='MI_value')
    annotator.set_pvalues([p])
    annotator.annotate()
plt.tight_layout()
plt.show()



# Step 2: Extract z-scored MI values
z_keys = [f'z_{key}' for key in ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']]
X = mi_pd_pyr[z_keys].values
# Step 3: Run KMeans clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(X)
mi_pd_pyr['cluster'] = cluster_labels
# Step 4: Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_embedding = tsne.fit_transform(X)
mi_pd_pyr['tsne_x'] = tsne_embedding[:, 0]
mi_pd_pyr['tsne_y'] = tsne_embedding[:, 1]
# Step 5: Prepare custom area colors
area_palette = {'SUPERFICIAL': '#800080', 'DEEP': '#FFD700'}  # purple and gold
# Step 6: Plotting
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.ravel()
# Plot 1: t-SNE by Cluster ID
sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster', data=mi_pd_pyr, ax=axs[0],
                palette='tab10', s=30, legend='brief')
axs[0].set_title('t-SNE by Cluster ID')
# Plot 2: t-SNE by Mouse
sns.scatterplot(x='tsne_x', y='tsne_y', hue='mouse', data=mi_pd_pyr, ax=axs[1],
                palette='Set2', s=30, legend='brief')
axs[1].set_title('t-SNE by Mouse')
# Plot 3: t-SNE by Area (with custom colors)
sns.scatterplot(x='tsne_x', y='tsne_y', hue='area', data=mi_pd_pyr, ax=axs[2],
                palette=area_palette, s=30, legend='brief')
axs[2].set_title('t-SNE by Area')
# Plot 4–10: t-SNE colored by raw MI values
mi_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']
for i, key in enumerate(mi_keys, start=3):
    sc = axs[i].scatter(mi_pd_pyr['tsne_x'], mi_pd_pyr['tsne_y'],
                        c=mi_pd_pyr[key], cmap='coolwarm', vmin=0, vmax=0.4, s=10)
    axs[i].set_title(f't-SNE colored by {key}')
    plt.colorbar(sc, ax=axs[i])
# Cleanup layout
for ax in axs:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()







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
from src.config import *
import pickle

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

# Setup
cluster_directory = os.path.join(base_directory, 'clusters')

files_names = ['Achilles_mi_transferred_cluster_3_dict.pkl',
               'Calb_mi_transferred_cluster_3_dict.pkl',
               'Thy_mi_transferred_cluster_3_dict.pkl']
mouse_names = ['Achilles', 'Calb', 'Thy']

files_names = ['Achilles_mi_transferred_cluster_3_dict.pkl']
mouse_names = ['Achilles']

#files_names = ['Calb_mi_transferred_cluster_3_dict.pkl',
#               'Thy_mi_transferred_cluster_3_dict.pkl']
#mouse_names = ['Calb', 'Thy']

def preprocess_spikes(traces, sig_up=6, sig_down=12):
    #lp_traces = uniform_filter1d(traces, size=4000, axis=0)
    #clean_traces = gaussian_filter1d(traces, sigma=sig_filt, axis=0)
    conv_traces = np.zeros(traces.shape)

    gaus = lambda x, sig, amp, vo: amp * np.exp(-(((x) ** 2) / (2 * sig ** 2))) + vo;
    x = np.arange(-5 * sig_down, 5 * sig_down, 1);
    left_gauss = gaus(x, sig_up, 1, 0);
    left_gauss[5 * sig_down + 1:] = 0
    right_gauss = gaus(x, sig_down, 1, 0);
    right_gauss[:5 * sig_down + 1] = 0
    gaus_kernel = right_gauss + left_gauss;

    for cell in range(traces.shape[1]):
        conv_traces[:, cell] = np.convolve(traces[:, cell], gaus_kernel, 'same')
    return conv_traces

behavior_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']

# Initialize storage
mouse_name_list, area_list, probe_list, typeID_list = [], [], [], []
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}

# Load data
speed_limit = {'Achilles': 0.05, 'Calb': 6, 'Thy': 6}
for fname, mouse in zip(files_names, mouse_names):
    filepath = os.path.join(cluster_directory, fname)
    with open(filepath, 'rb') as f:
        mi_dict = pickle.load(f)
    for probe, probe_data in mi_dict.items():
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
        valid_index = np.where(probe_data['behaviour']['speed'] > speed_limit[mouse])[0]
        position = beh_variables['pos'][valid_index]
        time = beh_variables['time'][valid_index]

        signal = signal[valid_index,:]
        clusters_names = np.unique(clusterID)
        random_neuron = 10
        trial_id = beh_variables['trial_id_mat'][valid_index]
        internal_time = compute_internal_trial_time(trial_id, 40)
        # Define the trial_ids we want to display
        trial_ids_to_show = [10, 11, 12, 13, 14, 15, 16]

        # Get the indices where trial_id is in the desired range
        trial_mask = np.isin(trial_id, trial_ids_to_show)
        index_filtered = trial_mask
        position_filtered = position[index_filtered]

        valid_clusters = [c for c in clusters_names if c != 10]
        fig, axs = plt.subplots(2 + len(valid_clusters), 1, figsize=(12, 18))
        time_filtered = np.arange(len(position_filtered))
        # --- 1) Position vs time
        axs[0].plot(time_filtered, position_filtered, linewidth=1.5)
        axs[0].set_title('Position vs Time')
        axs[0].set_xlabel('Time (samples)')
        axs[0].set_ylabel('Position')
        # --- 2) One cell with multiple Gaussian filters
        spikes_matrix_filtered = signal[index_filtered]
        sigma = 10
        n = min(random_neuron, spikes_matrix_filtered.shape[1] - 1)
        raw_traces = preprocess_spikes(spikes_matrix_filtered, sig_up=12, sig_down=24)
        raw_trace = raw_traces[:, n]
        sm = gaussian_filter1d(raw_trace.astype(float), sigma=sigma, axis=0, mode='nearest')
        axs[1].plot(time_filtered, raw_trace, label=f'raw', linewidth=1.25)
        axs[1].plot(time_filtered, sm, label=f'σ={sigma}', linewidth=1.25)
        axs[1].set_title(f'Neuron {n} smoothed (σ={sigma})')
        axs[1].set_xlabel('Time (samples)')
        axs[1].set_ylabel('Activity (a.u.)')
        axs[1].legend(loc='upper right', ncol=2, fontsize=9, frameon=False)
        # Convert spikes to rates (keeps your original call signature/shape use)
        rates = raw_traces
        index = 0
        for cluster_idx in clusters_names:
            if cluster_idx == 10:
                continue
            # neurons × time matrix for this cluster
            cluster_signal = raw_traces[:, clusterID == cluster_idx].T  # (neurons, time)
            if cluster_signal.size == 0:
                continue  # skip empty clusters gracefully
            # --- sort neurons by where their maximum occurs (earliest peaks first)
            # If NaNs can appear, uncomment the next line:
            # safe_signal = np.where(np.isnan(cluster_signal), -np.inf, cluster_signal)
            safe_signal = cluster_signal
            peak_times = np.argmax(safe_signal, axis=1)  # (neurons,)
            order = np.argsort(peak_times)  # ascending time of peak
            cluster_signal_sorted = cluster_signal[order, :]
            # --- plot with white->black (black = higher)
            im3 = axs[2 + index].imshow(
                cluster_signal_sorted,
                aspect='auto',
                origin='lower',  # earliest-peak neurons appear at the bottom
                cmap='gray_r'  # reversed gray: white->black, black = high
                # Optional robust scaling:
                # vmin=np.percentile(cluster_signal_sorted, 1),
                # vmax=np.percentile(cluster_signal_sorted, 99)
            )
            axs[2 + index].set_title(f'Cluster {cluster_idx}: neurons sorted by peak time')
            axs[2 + index].set_xlabel('Time (samples)')
            axs[2 + index].set_ylabel('Neuron (sorted)')
            # cbar3 = plt.colorbar(im3, ax=axs[2 + index], fraction=0.046, pad=0.04)
            #cbar3.set_label('Scaled activity')
            index += 1
        plt.tight_layout()
        plt.savefig(os.path.join(figures_directory, 'heatmap_cluster_example.svg'))
        plt.show()


####################################################33
random_neuron = 10
index_filtered = np.arange(0,len(position),1)
position_filtered = position[index_filtered]
fig, axs = plt.subplots(2 + len(clusters_names), 1, figsize=(12, 18))

time_filtered = np.arange(len(position_filtered))
# --- 1) Position vs time
axs[0].plot(time_filtered, position_filtered, linewidth=1.5)
axs[0].set_title('Position vs Time')
axs[0].set_xlabel('Time (samples)')
axs[0].set_ylabel('Position')
# --- 2) One cell with multiple Gaussian filters
spikes_matrix_filtered = signal[index_filtered]
sigma = 10
n = min(random_neuron, spikes_matrix_filtered.shape[1] - 1)
raw_trace = spikes_matrix_filtered[:, n]
sm = gaussian_filter1d(raw_trace.astype(float), sigma=sigma, axis=0, mode='nearest')
axs[1].plot(time_filtered, sm, label=f'σ={s}', linewidth=1.25)
axs[1].set_title(f'Neuron {n} smoothed σ')
axs[1].set_xlabel('Time (samples)')
axs[1].set_ylabel('Activity (a.u.)')
axs[1].legend(loc='upper right', ncol=2, fontsize=9, frameon=False)
rates = spikes_to_rates(spikes_matrix_filtered, sigma)
index = 0
for cluster_idx in clusters_names:
    if cluster_idx == 10: continue;
    cluster_signal = rates[:, clusterID == cluster_idx]
    im3 = axs[2+index].imshow(cluster_signal, aspect='auto', origin='lower')
    axs[2+index].set_title(f'')
    axs[2+index].set_xlabel('Time (samples)')
    axs[2+index].set_ylabel('Neuron (sorted)')
    cbar3 = plt.colorbar(im3, ax=axs[2+index], fraction=0.046, pad=0.04)
    cbar3.set_label('Scaled activity')
    index += 1


plt.tight_layout()
plt.savefig(os.path.join(figures_directory + 'heatmap_cluster_example.svg'))
plt.show()

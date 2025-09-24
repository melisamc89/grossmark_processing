

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


### adapted from Julio's proprocess_traces

def preprocess_spikes(traces, sig_up=8, sig_down=24):
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

rats = [0]
sessions = [[0],[0]]
speed_lim = 0.05 #(m/s)
for rat_index in rats:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1','Probe2']:
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

            invalid_mov = np.logical_and(speed < speed_lim, speed <speed_lim)
            valid_mov = np.logical_not(invalid_mov)

            position = position[valid_mov]
            speed = speed[valid_mov]
            direction = direction[valid_mov]
            time_stamps = time_stamps[valid_mov]
            trial_id = get_trial_id(position)
            internal_time = compute_internal_trial_time(trial_id, 40)
            time = time_stamps

            beh_variables = {
                'pos': position,
                'dir': direction,
                'time': time
            }

            spikes_matrix = stimes['spikes_matrix']
            spikes_matrix = spikes_matrix[valid_mov,:]
            from scipy.signal import find_peaks
            import matplotlib.pyplot as plt
            import numpy as np
            random_neuron = 10
            # Define the trial_ids we want to display
            trial_ids_to_show = [1,2,3,4,5,6,7,8,9,10, 11, 12]

            # Get the indices where trial_id is in the desired range
            trial_mask = np.isin(trial_id, trial_ids_to_show)

            # Apply the mask to filter position, speed, direction, and other signals
            position_filtered = position[trial_mask]
            time_filtered = time[trial_mask]
            speed_filtered = speed[trial_mask]
            direction_filtered = direction[trial_mask]
            spikes_matrix_filtered = spikes_matrix[trial_mask, :]

            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d

            fig, axs = plt.subplots(4, 1, figsize=(10, 20))
            time_filtered = np.arange(0, len(position_filtered))
            # Plot position
            axs[0].plot(time_filtered, position_filtered, label='Position', color='k')
            # Filtered signal with sigma=20
            filtered_signal_20 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=20)
            norm_filtered_20 = (filtered_signal_20 - np.min(filtered_signal_20)) / (
                        np.max(filtered_signal_20) - np.min(filtered_signal_20) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 1000 * norm_filtered_20,
                c='yellow',
                alpha=norm_filtered_20,
                label='Filtered σ=20'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=10)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 500 * norm_filtered_10,
                c='green',
                alpha=norm_filtered_10,
                label='Filtered σ=10'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=5)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 250 * norm_filtered_10,
                c='blue',
                alpha=norm_filtered_10,
                label='Filtered σ=5'
            )
            # Filtered signal with sigma=1
            peaks, _ = find_peaks(spikes_matrix_filtered[:, random_neuron])
            axs[0].scatter(
                time_filtered[peaks],
                position_filtered[peaks],
                s=100,
                c='k',
                alpha=1,
                label='Peaks'
            )
            axs[0].set_title('Position Over Time (Trial ID = 10, 11, 12)')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Position (m)')
            axs[0].set_xlim([0, len(position_filtered)])
            axs[0].legend()
            # Heatmaps
            filtered_spikes_1 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=1, axis=1)
            cax1 = axs[1].imshow(filtered_spikes_1, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[1].set_title('Heatmap (Filter = 1 ms)')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Neurons')
            axs[1].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[1].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_10 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=10, axis=1)
            cax2 = axs[2].imshow(filtered_spikes_10, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[2].set_title('Heatmap (Filter = 10 ms)')
            axs[2].set_xlabel('Time (ms)')
            axs[2].set_ylabel('Neurons')
            axs[2].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[2].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_20 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=20, axis=1)
            cax3 = axs[3].imshow(filtered_spikes_20, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[3].set_title('Heatmap (Filter = 20 ms)')
            axs[3].set_xlabel('Time (ms)')
            axs[3].set_ylabel('Neurons')
            axs[3].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[3].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory + 'heatmap_example.svg'))
            plt.show()



fig, axs = plt.subplots(4, 1, figsize=(12, 18))
time_filtered = np.arange(len(position_filtered))

# --- 1) Position vs time
axs[0].plot(time_filtered, position_filtered, linewidth=1.5)
axs[0].set_title('Position vs Time')
axs[0].set_xlabel('Time (samples)')
axs[0].set_ylabel('Position')

# --- 2) One cell with multiple Gaussian filters

sigmas = [4, 8, 12, 16]
n = min(random_neuron, spikes_matrix_filtered.shape[1] - 1)

raw_traces = preprocess_spikes(spikes_matrix_filtered)
raw_trace = raw_traces[:, n]

for s in sigmas:
    sm = gaussian_filter1d(raw_trace.astype(float), sigma=s, axis=0, mode='nearest')
    axs[1].plot(time_filtered, sm, label=f'σ={s}', linewidth=1.25)

axs[1].set_title(f'Neuron {n} smoothed with multiple σ')
axs[1].set_xlabel('Time (samples)')
axs[1].set_ylabel('Activity (a.u.)')
axs[1].legend(loc='upper right', ncol=2, fontsize=9, frameon=False)

# --- Helpers for heatmaps
def smooth_and_scale(mat, sigma):
    """Gaussian smooth across time and scale each neuron to [0,1] for visualization."""
    sm = gaussian_filter1d(mat.astype(float), sigma=sigma, axis=0, mode='nearest')
    # Min-max per neuron to emphasize temporal structure in the heatmap
    mn = sm.min(axis=0, keepdims=True)
    mx = sm.max(axis=0, keepdims=True)
    # Avoid divide-by-zero
    scaled = (sm - mn) / np.maximum(mx - mn, 1e-12)
    return sm, scaled

# --- 3) Heatmap (σ=8), neurons sorted by time of max activity (on σ=8)
sigma_sort_1 = 8
sm8, sc8 = smooth_and_scale(spikes_matrix_filtered, sigma=sigma_sort_1)
tmax8 = np.argmax(sm8, axis=0)             # time-of-peak for each neuron
order8 = np.argsort(tmax8)                  # neurons sorted by peak time
H8 = sc8[:, order8].T                       # (neurons x time)

im3 = axs[2].imshow(H8, aspect='auto', origin='lower',
                    extent=[time_filtered[0], time_filtered[-1], 0, H8.shape[0]])
axs[2].set_title(f'All neurons heatmap (σ={sigma_sort_1}), sorted by peak time (σ={sigma_sort_1})')
axs[2].set_xlabel('Time (samples)')
axs[2].set_ylabel('Neuron (sorted)')
cbar3 = plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
cbar3.set_label('Scaled activity')

# --- 4) Heatmap (σ=20), neurons sorted by time of max activity (on σ=20)
sigma_sort_2 = 20
sm20, sc20 = smooth_and_scale(spikes_matrix_filtered, sigma=sigma_sort_2)
tmax20 = np.argmax(sm20, axis=0)
order20 = np.argsort(tmax20)
H20 = sc20[:, order20].T

im4 = axs[3].imshow(H20, aspect='auto', origin='lower',
                    extent=[time_filtered[0], time_filtered[-1], 0, H20.shape[0]])
axs[3].set_title(f'All neurons heatmap (σ={sigma_sort_2}), sorted by peak time (σ={sigma_sort_2})')
axs[3].set_xlabel('Time (samples)')
axs[3].set_ylabel('Neuron (sorted)')
cbar4 = plt.colorbar(im4, ax=axs[3], fraction=0.046, pad=0.04)
cbar4.set_label('Scaled activity')

plt.tight_layout()
plt.savefig(os.path.join(figures_directory + 'heatmap_example_2.svg'))
plt.show()

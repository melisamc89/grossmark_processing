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
from scipy.signal import welch, decimate
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly
from src.structure_index import *

def compute_power_spectrum_in_lfp(shank_signal, fs):
    power_sectrum = {}
    for shank in shank_signal.keys():
        signal = shank_signal[shank]['lfp']
        n_samples = signal.shape[0]
        freqs = np.fft.fftfreq(n_samples, d=1 / fs)
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        power_spectra = np.zeros(((len(freqs), signal.shape[1])))
        for i in range(signal.shape[1]):
            # Compute the FFT for each column
            fft_result = np.fft.fft(signal[:, i])
            # Compute the power spectrum
            power = np.abs(fft_result) ** 2
            power_spectra[:,i] = power[positive_mask]  # Consider only the positive part of the spectrum
        power_sectrum[shank] = {'frequency': freqs, 'power': power_spectra}
    return power_sectrum

def compute_band_power_in_shanks_probe(power_spectrum, low_freq, high_freq):
    bands_power = {}
    for shank_id in power_spectrum.keys():
        freqs = power_spectrum[shank_id]['frequency']
        power =power_spectrum[shank_id]['power']
        # Find the indices where frequency is within the specified range
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        # Sum the power values within this range
        power = power/np.max(power, axis = 0)
        band_power_sum = np.sum(power[band_mask],axis=0)
        bands_power[shank_id] = band_power_sum
    return bands_power

def find_max_power_in_shanks(power_dict):
    max_positions = {}
    for shank, values in power_dict.items():
        # Find the index of the maximum value
        max_index = np.argmax(values)
        max_positions[shank] = max_index
    return max_positions

def find_max_value_and_index(data_dict):
    max_value = -np.inf
    max_shank = None
    max_index = -1
    for shank, values in data_dict.items():
        # Find the index of the maximum value in the current shank
        current_index = np.argmax(values)
        current_max_value = values[current_index]
        if current_max_value > max_value:
            max_value = current_max_value
            max_shank = shank
            max_index = current_index
        # Store the maximum value's shank and index
    max_values = {'shank': max_shank, 'index': max_index}
    return max_values

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

# Your bands
bands = {
    'infra-slow':(0.01,0.1),
    'delta': (1, 3),
    'theta': (8, 12),
    'slow-gamma': (40, 90),
    'ripple-band': (100, 250),
    'MUA': (300, 500)
}

from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis = 0)
novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')

neural_data_dir = files_directory
lfp_dir = lfp_directory

rat_index = [0,1,2,3]
sessions = [[0],[0],[2],[1]]
speed_lim = 0.05

params1 = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 0,
    'verbose': False
}
si_neigh = 50
si_lfp_params = {}
for band in bands.keys():
    si_lfp_params[band] = copy.deepcopy(params1)

for rat_index in [0]:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        file_name_lfp = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
            session_index] + '_lfp_output.pkl'
        lfp_file_dir = os.path.join(lfp_dir, file_name_lfp)
        with open(lfp_file_dir, 'rb') as file:
            # Load the data from the file
            lfp = pkl.load(file)

        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)
        session_info = convert_session_mat_to_dict(session_information_directory)
        start_maze = session_info['epochs']['mazeEpoch'][0][0]
        end_maze = session_info['epochs']['mazeEpoch'][1][0]

        preposition1D = session_info['position']['position1D']
        not_nan = ~np.isnan(preposition1D)
        position1D = preposition1D[not_nan].reshape(-1)
        positionTimeStamps = np.hstack(session_info['position']['time_staps']) - start_maze
        positionTimeStamps= positionTimeStamps[not_nan[0]]

        for probe in ['Probe1','Probe2']:
            si_dict= dict()
            lfp_probe = lfp[probe]
            # Parameters
            power_spectrum = compute_power_spectrum_in_lfp(lfp_probe, 1250)
            target_fs = 1250
            low_freq = 8
            high_freq = 12
            theta_power = compute_band_power_in_shanks_probe(power_spectrum, low_freq, high_freq)
            del power_spectrum
            max_theta_shanks = find_max_power_in_shanks(theta_power)
            max_theta = find_max_value_and_index(theta_power)
            lfp_max_theta = lfp_probe[max_theta['shank']]['lfp'][:,max_theta['index']]
            del lfp_probe
            # Filter and downsample signals
            from scipy.signal import butter, filtfilt

            fs_raw = 1250  # Original LFP sampling rate

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

            invalid_mov = np.logical_or(speed < speed_lim, direction == 0)
            valid_mov = np.logical_not(invalid_mov)

            position = position[valid_mov]
            speed = speed[valid_mov]
            direction = direction[valid_mov]
            time_stamps = time_stamps[valid_mov]
            trial_id = trial_id[valid_mov]
            internal_time = internal_time[valid_mov]
            time = time[valid_mov]

            filtered_downsampled = {}
            for band_name, (low, high) in bands.items():
                if high >= 0.5 * fs_raw:
                    print(f"Skipping {band_name}: upper bound too close to Nyquist.")
                    continue
                # Bandpass filter at 1250 Hz
                filtered = bandpass_filter(lfp_max_theta, low, high, fs=fs_raw)
                # Convert timestamps (in seconds) to sample indices
                sample_indices = np.round(positionTimeStamps * fs_raw).astype(int)
                # Prevent indexing out of bounds
                sample_indices = sample_indices[sample_indices < len(filtered)]
                # Extract the signal at those sample points
                aux = filtered[sample_indices]
                filtered_downsampled[band_name] = aux[valid_mov]

            k = 20
            spikes_matrix = stimes['spikes_matrix']
            spikes_matrix = spikes_matrix[valid_mov,:]
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)

            typeID = stimes['TypeID']
            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])
            data=data[:,pyr_index]
            umap_model = umap.UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)
            umap_model.fit(data)
            umap_emb = umap_model.transform(data)

            bands_range = {'infra-slow': [0,100],
                        'delta':[0,100],
                           'theta':[0,600],
                           'slow-gamma':[50,100],
                           'ripple-band':[0,60],
                           'MUA':[0,30]}

            from scipy.ndimage import gaussian_filter1d

            si, process_info, overlap_mat, _ = compute_structure_index(data, position,
                                                                       **si_lfp_params['delta'])
            si_umap, process_info, overlap_mat, _ = compute_structure_index(umap_emb, position,
                                                                       **si_lfp_params['delta'])
            print(f" position ={si:.4f} |", end='', sep='', flush='True')
            print(f" position_umap ={si_umap:.4f} |", end='', sep='', flush='True')
            print()

            si, process_info, overlap_mat, _ = compute_structure_index(data, direction,
                                                                       **si_lfp_params['delta'])
            si_umap, process_info, overlap_mat, _ = compute_structure_index(umap_emb, direction,
                                                                       **si_lfp_params['delta'])
            print(f" direction ={si:.4f} |", end='', sep='', flush='True')
            print(f" direction_umap ={si_umap:.4f} |", end='', sep='', flush='True')
            print()

            si, process_info, overlap_mat, _ = compute_structure_index(data, time,
                                                                       **si_lfp_params['delta'])
            si_umap, process_info, overlap_mat, _ = compute_structure_index(umap_emb, time,
                                                                       **si_lfp_params['delta'])
            print(f" time ={si:.4f} |", end='', sep='', flush='True')
            print(f" time_umap ={si_umap:.4f} |", end='', sep='', flush='True')
            print()

            sigma = 3  # Gaussian filter smoothing parameter
            # First 6 plots: band power with Gaussian smoothing
            for index, band_name in enumerate(bands):
                band_signal = filtered_downsampled[band_name]
                smoothed_signal = gaussian_filter1d(np.abs(band_signal), sigma=sigma)
                si_lfp_params[band_name]['n_neighbors'] = si_neigh
                si_dict[band_name] = dict()

                si, process_info, overlap_mat, _ = compute_structure_index(data,band_signal,
                                                                            **si_lfp_params[band_name])

                si_umap, process_info, overlap_mat, _ = compute_structure_index(umap_emb,band_signal,
                                                                            **si_lfp_params[band_name])



                si_dict[band_name] = {
                    'si': copy.deepcopy(si),
                    'si_umap': copy.deepcopy(si_umap),
                }

                print(f" {band_name} ={si:.4f} |", end='', sep='', flush='True')
                print(f" {band_name} _umap ={si_umap:.4f} |", end='', sep='', flush='True')

                print()

            data_output_directory = '/home/melma31/Documents/time_project/SI_bands'
            # Define the filename where the dictionary will be stored
            output_filename = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] +'_'+ probe + '_si_bands.pkl'
            # Open the file for writing in binary mode and dump the dictionary
            with open(os.path.join(data_output_directory, output_filename), 'wb') as file:
                pkl.dump(si_dict, file)



import matplotlib.pyplot as plt
import numpy as np

# Define the data
labels_1 = ['position', 'direction', 'time']
data1_1 = [0.8290, 0.9317, 0.9732]
data2_1 = [0.7748, 0.9021, 0.7113]
data1_1_umap = [0.8559, 0.9385, 0.8097]
data2_1_umap = [0.8088, 0.9027, 0.6656]

labels_2 = ['infra-slow', 'delta', 'theta', 'slow-gamma', 'ripple-band', 'MUA']
data1_2 = [0.4109, 0.0485, 0.0144, 0.0001, 0.0000, 0.0000]
data2_2 = [0.4289, 0.0433, 0.0154, 0.0005, 0.0000, 0.0000]
data1_2_umap = [0.3569, 0.0488, 0.0168, 0.0000, 0.0002, 0.0000]
data2_2_umap = [0.3457, 0.0462, 0.0127, 0.0000, 0.0009, 0.0000]

# Compute mean and std for each set
def compute_stats(d1, d2):
    mean = np.mean([d1, d2], axis=0)
    std = np.std([d1, d2], axis=0)
    return mean, std

mean_1, std_1 = compute_stats(data1_1, data2_1)
mean_1_umap, std_1_umap = compute_stats(data1_1_umap, data2_1_umap)

mean_2, std_2 = compute_stats(data1_2, data2_2)
mean_2_umap, std_2_umap = compute_stats(data1_2_umap, data2_2_umap)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
# Subplot 1: position, direction, time
x = np.arange(len(labels_1))
width = 0.35
axs[0].bar(x - width/2, mean_1, width, yerr=std_1, capsize=5, label='Traces')
axs[0].bar(x + width/2, mean_1_umap, width, yerr=std_1_umap, capsize=5, label='UMAP')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels_1)
axs[0].set_ylabel('Value')
axs[0].set_title('High-level Features')
axs[0].legend()

# Subplot 2: remaining bands
x2 = np.arange(len(labels_2))
axs[1].bar(x2 - width/2, mean_2, width, yerr=std_2, capsize=5, label='Raw')
axs[1].bar(x2 + width/2, mean_2_umap, width, yerr=std_2_umap, capsize=5, label='UMAP')
axs[1].set_xticks(x2)
axs[1].set_xticklabels(labels_2, rotation=45)
axs[1].set_ylabel('Value')
axs[1].set_title('Oscillatory Bands & MUA')
axs[1].legend()

plt.tight_layout()
plt.show()

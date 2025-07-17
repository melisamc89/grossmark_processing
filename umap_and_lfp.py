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


            k = 10
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

            from scipy.stats import zscore

            bands_range = {'infra-slow': [0,300],
                        'delta':[0,100],
                           'theta':[0,800],
                           'slow-gamma':[0,200],
                           'ripple-band':[0,100],
                           'MUA':[0,50]}

            row, col = 3, 3
            fig = plt.figure(figsize=(12, 14))

            # First 6 plots: band power
            for index, band_name in enumerate(bands):
                band_signal = filtered_downsampled[band_name]

                if band_signal is None or len(band_signal) != umap_emb.shape[0]:
                    print(f"Skipping {band_name} due to shape mismatch.")
                    continue

                color_vals = abs(band_signal)
                ax = fig.add_subplot(row, col, index + 1, projection='3d')
                ax.set_title(band_name)
                vmin = bands_range[band_name][0]
                vmax = bands_range[band_name][1]
                scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2],
                                     c=color_vals, s=2, alpha=0.5, cmap='jet', vmin=vmin, vmax=vmax)
                ax.grid(False)

            # Plot UMAP colored by position
            ax = fig.add_subplot(row, col, 7, projection='3d')
            ax.set_title('Position')
            scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2],
                                 c=position, cmap='magma', s=2, alpha=0.5)
            ax.grid(False)

            # Plot UMAP colored by direction
            ax = fig.add_subplot(row, col, 8, projection='3d')
            ax.set_title('Direction')
            scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2],
                                 c=direction, cmap='Blues', s=2, alpha=0.5)
            ax.grid(False)

            # Plot UMAP colored by time
            ax = fig.add_subplot(row, col, 9, projection='3d')
            ax.set_title('Time')
            scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2],
                                 c=time, cmap='YlGn_r', s=2, alpha=0.5)
            ax.grid(False)

            plt.tight_layout()
            plt.show()

            # Define the filename where the dictionary will be stored
            figure_name = rat_names[rat_index] + '_' + str(rat_sessions[rat_names[rat_index]][session_index]) +'_' +probe + 'umap_lfp.png'
            fig.savefig(os.path.join(figures_directory, figure_name))



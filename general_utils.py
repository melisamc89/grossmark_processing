import xml.etree.ElementTree as ET
import scipy as sp
import h5py
from scipy.ndimage import gaussian_filter1d


#### funcions to move later
def get_recorging_parameters(file_name):
    # Load and parse the XML file
    ### add check if file exist and check if it is xml file!
    tree = ET.parse(file_name)
    root = tree.getroot()
    # Print the root element tag
    # Iterate through child elements of the root
    for child in root:
        # Accessing specific child elements or data
        for subchild in child:
            if subchild.tag == 'nChannels':
                nChannels = int(subchild.text)
            if subchild.tag == 'lfpSamplingRate':
                lfp_sr = float(subchild.text)
            if subchild.tag == 'samplingRate':
                sr = float(subchild.text)

    params = {}
    params['nChannels'] = nChannels
    params['lfpSamplingRate'] = lfp_sr
    params['sr'] = sr
    return params


def convert_session_mat_to_dict(mat_file_path):

    session_info = {}
    with h5py.File(mat_file_path, 'r') as file:
        sess_info = file['sessInfo']
        # Assuming 'Spikes' is a subgroup under 'sessInfo'
        spikes_group = sess_info['Spikes']
        spike_times = np.array(spikes_group['SpikeTimes'])
        spike_ids = np.array(spikes_group['SpikeIDs'])
        pyr_ids = np.array(spikes_group['PyrIDs'])
        int_ids = np.array(spikes_group['IntIDs'])
        spikes_dict = {}
        spikes_dict['spikes_times'] = spike_times
        spikes_dict['spikes_ids'] = spike_ids
        spikes_dict['pyr_ids'] = pyr_ids
        spikes_dict['int_ids'] = int_ids
        session_info['spikes'] = spikes_dict

        position_group = sess_info['Position']
        position2D = np.array(position_group['TwoDLocation'])
        position1D = np.array(position_group['OneDLocation'])
        time_staps = np.array(position_group['TimeStamps'])
        maze_type = np.array(position_group['MazeType'])
        position_dict = {}
        position_dict['position2D'] = position2D
        position_dict['position1D'] = position1D
        position_dict['time_staps'] = time_staps
        position_dict['maze_type'] = maze_type
        session_info['position'] = position_dict

        epochs_group = sess_info['Epochs']
        preEpoch = np.array(epochs_group['PREEpoch'])
        mazeEpoch = np.array(epochs_group['MazeEpoch'])
        postEpoch = np.array(epochs_group['POSTEpoch'])
        sessDur = np.array(epochs_group['sessDuration'])
        WakeEpoch = np.array(epochs_group['Wake'])
        DrowsyEpoch = np.array(epochs_group['Drowsy'])
        NREMEpoch = np.array(epochs_group['NREM'])
        IntermediateEpoch = np.array(epochs_group['Intermediate'])
        REMEpoch = np.array(epochs_group['REM'])
        epochs_dict = {}
        epochs_dict['preEpoch'] = preEpoch
        epochs_dict['mazeEpoch'] = mazeEpoch
        epochs_dict['postEpoch'] = postEpoch
        epochs_dict['sessDur'] = sessDur
        epochs_dict['WakeEpoch'] = WakeEpoch
        epochs_dict['DrowsyEpoch'] = DrowsyEpoch
        epochs_dict['NREMEpoch'] = NREMEpoch
        epochs_dict['IntermidiateEpoch'] = IntermediateEpoch
        epochs_dict['REMEpoch'] = REMEpoch
        session_info['epochs'] = epochs_dict

    return session_info

def group_lfp_by_shank(lfp_data, channel_dict):
    shank_signals = {}
    for probes in channel_dict.keys():
        probes_signals = {}
        for shanks in channel_dict[probes].keys():
            if 'Spk.Group' in shanks:  # Skip Spk.Group entries
                    continue
            # Adjust channel indices for Python's 0-based indexing
            channels = channel_dict[probes][shanks]
            channel_indices = [ch - 1 for ch in channels]
            # Calculate mean signal across specified channels
            signal = lfp_data[:, channel_indices]
            # Store in dictionary
            shank_id = f"{shanks}"
            probes_signals[shank_id] = signal
        probe_id = f"{probes}"
        shank_signals[probe_id] = probes_signals
    return shank_signals

def get_signals_segment(signals, start, end):
    segment_signals = signals[start:end]
    return segment_signals

import scipy as sp
import numpy as np

def downsample_signal(signal, original_fs, new_fs):
    # Number of original samples
    num_original_samples = signal.shape[0]

    # Calculate the new number of samples
    num_new_samples = int(num_original_samples * (new_fs / original_fs))

    # Resample the signal
    resampled_signal = sp.signal.resample(signal, num_new_samples, axis=0)

    return resampled_signal


def compute_power_spectrum_in_shanks(shank_signal, fs):
    power_sectrum = {}
    for probe in shank_signal.keys():
        probe_power = {}
        for shank in shank_signal[probe].keys():
            signal = shank_signal[probe][shank]

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
            probe_power[shank] = {'frequency': freqs, 'power': power_spectra}
        power_sectrum[probe] = probe_power
    return power_sectrum

def compute_band_power_in_shanks(power_spectrum, low_freq, high_freq):
    bands_power = {}
    for probe in power_spectrum.keys():
        bands_power[probe] = {}
        for shank_id in power_spectrum[probe].keys():
            freqs = power_spectrum[probe][shank_id]['frequency']
            power =power_spectrum[probe][shank_id]['power']
            # Find the indices where frequency is within the specified range
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            # Sum the power values within this range
            power = power/np.max(power, axis = 0)
            band_power_sum = np.sum(power[band_mask],axis=0)
            bands_power[probe][shank_id] = band_power_sum
    return bands_power

def find_max_power_in_shanks(power_dict):
    max_positions = {}
    for probe, shanks in power_dict.items():
        max_positions[probe] = {}
        for shank, values in shanks.items():
            # Find the index of the maximum value
            max_index = np.argmax(values)
            max_positions[probe][shank] = max_index
    return max_positions

def find_max_value_and_index(data_dict):
    max_values = {}
    for probe, shanks in data_dict.items():
        # Initialize variables to track the maximum value and corresponding shank and index
        max_value = -np.inf
        max_shank = None
        max_index = -1
        for shank, values in shanks.items():
            # Find the index of the maximum value in the current shank
            current_index = np.argmax(values)
            current_max_value = values[current_index]
            # Check if the current maximum is greater than the overall maximum found so far
            if current_max_value > max_value:
                max_value = current_max_value
                max_shank = shank
                max_index = current_index
        # Store the maximum value's shank and index
        max_values[probe] = {'shank': max_shank, 'index': max_index}
    return max_values

def downsample_spectrum_in_shanks(power_spectrum,factor):
    downsampled_power_spectrum = {}
    for probe in power_spectrum.keys():
        downsampled_power_spectrum[probe] = {}
        for shank in power_spectrum[probe].keys():
            downsampled_power_spectrum[probe][shank] = {}
            # Number of complete bins in the downsampled spectrum
            downsample_matrix = np.zeros((power_spectrum[probe][shank]['power'].shape[0]//factor,power_spectrum[probe][shank]['power'].shape[1]))
            for i in range(power_spectrum[probe][shank]['power'].shape[1]):
                power = power_spectrum[probe][shank]['power'][:,i]
                freqs = power_spectrum[probe][shank]['frequency']
                num_bins = len(power) // factor
                # Downsample by reshaping and averaging
                downsampled_power = power[:num_bins*factor].reshape(-1, factor).mean(axis=1)
                downsample_matrix[:,i] = downsampled_power
            downsampled_freqs = freqs[:num_bins*factor].reshape(-1, factor).mean(axis=1)
            downsampled_power_spectrum[probe][shank]['power'] = downsample_matrix
            downsampled_power_spectrum[probe][shank]['frequency'] = downsampled_freqs
    return downsampled_power_spectrum


def read_spike_waveforms(filename, num_channels, samples_per_spike=32):
    """
    Reads a .spk file containing spike waveforms.

    Args:
    filename (str): The path to the .spk file.
    num_channels (int): Number of channels recorded on the shank.
    samples_per_spike (int): Number of samples per spike waveform.

    Returns:
    numpy.ndarray: A 3D array where each slice [i, :, :] represents the waveform of the i-th spike
                   across all channels, with each row representing a channel.
    """
    data_type = np.int16
    # Calculate the total number of samples per spike across all channels
    total_samples_per_spike = samples_per_spike * num_channels
    # Read the file
    data = np.fromfile(filename, dtype=data_type)
    # Reshape the data. The number of spikes is inferred from the size of the file.
    waveforms =  data.reshape(-1,samples_per_spike, num_channels)
    return waveforms


def read_klusters_clu_file(filename):
    """
    Reads a .clu file generated by Klusters which contains cluster IDs for spikes.
    Args:
    filename (str): Path to the .clu file.

    Returns:
    tuple:
        num_clusters (int): The number of clusters, including the noise cluster.
        cluster_ids (numpy.ndarray): Array of cluster IDs for each spike.
    """
    # Read the entire file into a NumPy array of int32
    cluster_ids = np.loadtxt(filename)
    # The first element is the number of clusters
    num_clusters = cluster_ids[0]
    # The rest of the elements are the cluster assignments for each spike
    return num_clusters, cluster_ids[1:]

def read_spikes_times(filename):
    # Read the entire file into a NumPy array of int32
    spikes_times = np.loadtxt(filename)
    return spikes_times

def get_trial_id(labels):
    signal = labels
    # Thresholds
    upper_threshold = 1.5
    lower_threshold = 0.1

    # Array to store cycle IDs
    cycle_ids = np.zeros_like(signal, dtype=int)

    # Variables to keep track of the state and current cycle ID
    current_cycle = 0
    cycle_started = False
    above_threshold = False

    # Iterate over the signal
    for i in range(len(signal)):
        if not cycle_started and signal[i] < lower_threshold:
            # Check if the signal is around zero to start a cycle
            cycle_started = True
            current_cycle += 1

        if cycle_started:
            # Check if the signal has crossed the upper threshold
            if signal[i] > upper_threshold:
                above_threshold = True

            # Check if the signal returns to around zero and it was above the threshold
            if above_threshold and signal[i] < lower_threshold:
                cycle_started = False  # End of cycle
                above_threshold = False  # Reset for the next cycle

        # Assign the current cycle ID
        cycle_ids[i] = current_cycle
    return cycle_ids


def get_directions(signal, low_threshold=0.1, high_threshold=1.5):
    """
    Assigns a direction to each point in the signal based on threshold crossings.

    Parameters:
    - signal: np.array, the input signal data.
    - low_threshold: float, the lower boundary for detecting upward trends.
    - high_threshold: float, the upper boundary for detecting downward trends.

    Returns:
    - directions: np.array, an array containing 1 for upward trend, -1 for downward trend,
                  and 0 for no clear direction.
    """
    # Initialize directions with zeros
    directions = np.zeros_like(signal)

    # Track the current direction
    current_direction = 0

    # Iterate through the signal
    for i in range(1, len(signal)):
        if signal[i] < high_threshold and signal[i - 1] >= high_threshold:
            current_direction = -1
        elif signal[i] > low_threshold and signal[i - 1] <= low_threshold:
            current_direction = 1
        if signal[i] < low_threshold:
            current_direction = 0
        if signal[i] > high_threshold:
            current_direction = 0

        # Assign the current direction
        directions[i] = current_direction

    return directions



def get_speed(signal, time_stamps, filter_size=10):
    """
    Computes the derivative of the signal (speed) and applies a Gaussian filter.

    Parameters:
    - signal: np.array, the input signal data.
    - filter_size: int, the sigma for the Gaussian kernel, controlling the amount of smoothing.

    Returns:
    - filtered_speed: np.array, the speed of the signal, smoothed, and same length as input signal.
    """
    # Calculate the speed as the difference between consecutive elements
    speed_pos = np.diff(signal)
    timediff = np.diff(time_stamps)
    speed = speed_pos / timediff
    # To make speed the same length as signal, append the last computed difference to the end
    speed = np.append(speed, speed[-1])

    # Apply a Gaussian filter to the speed
    filtered_speed = gaussian_filter1d(speed, sigma=filter_size)

    return filtered_speed


def compute_internal_trial_time(trial_ids, sampling_rate):
    """
    Computes an internal trial time that resets to zero at the start of each new trial.

    Parameters:
    - trial_ids: np.array, an array of trial IDs where an increment indicates a new trial.
    - sampling_rate: float, the rate at which data is sampled per second.

    Returns:
    - trial_time: np.array, a time array that resets to zero at the beginning of each trial.
    """
    # Initialize the trial time array
    trial_time = np.zeros_like(trial_ids, dtype=float)

    # Calculate time step per sample
    time_step = 1 / sampling_rate

    # Track the start of a new trial
    current_trial_id = trial_ids[0]
    last_reset_index = 0

    # Iterate through the trial IDs
    for i in range(1, len(trial_ids)):
        if trial_ids[i] != current_trial_id:
            # A new trial has started
            current_trial_id = trial_ids[i]
            last_reset_index = i
        # Calculate time since the last trial start
        trial_time[i] = (i - last_reset_index) * time_step

    return trial_time

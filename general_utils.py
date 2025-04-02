import xml.etree.ElementTree as ET
import scipy as sp
import h5py


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
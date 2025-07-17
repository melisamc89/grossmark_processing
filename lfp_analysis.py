from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import*

from scipy.signal import butter, filtfilt, hilbert, welch

# === Filtering functions ===
def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


for rat_index in [0]:
    low_freq = low_ripple_freq[rat_index]
    high_freq = high_ripple_freq[rat_index]
    print('Extraction Ripple Bands from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        eef_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.eeg'
        xml_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.xml'
        eeg_file_directory = os.path.join(rat_directory, session_directory,eef_file_name)
        xml_file_directory = os.path.join(rat_directory, session_directory,xml_file_name)
        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        session_info = convert_session_mat_to_dict(session_information_directory)
        params = get_recorging_parameters(xml_file_directory)

        #### start loading
        lfp_sr = params['lfpSamplingRate']
        lfp = np.fromfile(eeg_file_directory, dtype='int16')
        lfp = lfp.reshape((-1, number_of_channels[rat_names[rat_index]]))

        ### get NREM period
        NREM_durections = session_info['epochs']['NREMEpoch'][1,:] - session_info['epochs']['NREMEpoch'][0,:]
        NREM_max_duration = np.max(NREM_durections)
        NREM_max_duration_index = np.argmax(NREM_durections)

        start_resting = int(session_info['epochs']['NREMEpoch'][0,NREM_max_duration_index] * lfp_sr )
        end_resting = int(session_info['epochs']['NREMEpoch'][1,NREM_max_duration_index] * lfp_sr )
        NREM_signal = get_signals_segment(lfp, start_resting, end_resting)

        del lfp
        target_fs = 1000  # Target sampling rate in Hz
        downsampled_lfp= downsample_signal(NREM_signal, lfp_sr, target_fs)
        del NREM_signal

        # Accessing data for Rat A, Session 1, Probe 1, Shank 2
        channels = channel_organization[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][session_index]]
        #### process channels and signals and extract resting period
        shank_signals = group_lfp_by_shank(downsampled_lfp, channels)
        del downsampled_lfp
        ################################################
        ####        PLOTTING SINGALS #########
        ################################################
        ### only plot for short signals
        #plot_probe_signals(shank_signals)
        #################################################
        #### COMPUTE POWER SPECTRA (and plot)    ########
        #################################################
        power_spectrum = compute_power_spectrum_in_shanks(shank_signals, target_fs)
        ##### downsampled power spectrum
        downsampled_power_spectrum = downsample_spectrum_in_shanks(power_spectrum,1000)
        figure = plot_power_spectra(downsampled_power_spectrum)
        figure.savefig(figures_directory + 'power_spectrum_' + rat_names[rat_index] +
                       '_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.png')
        #################################################
        ##### combute ripple and theta bands total power
        low_freq = 100
        high_freq = 250
        ripple_power = compute_band_power_in_shanks(downsampled_power_spectrum,low_freq,high_freq)
        #low_freq = 6
        #high_freq = 12
        #theta_power = compute_band_power_in_shanks(downsampled_power_spectrum,low_freq,high_freq)
        spk_group = {}
        spk_group_probe_1 = channels['Probe1']['Spk.Group']
        spk_group_probe_2 = channels['Probe2']['Spk.Group']
        spk_group['Probe1'] = spk_group_probe_1
        spk_group['Probe2'] = spk_group_probe_2


        #max_ripple_shanks = find_max_power_in_shanks(ripple_power)
        #max_theta_shanks = find_max_power_in_shanks(theta_power)

        combined_dict = {}
        for probe in spk_group:
            combined_dict[probe] = {}
            for shank in spk_group[probe]:
                raw = shank_signals[probe][shank]
                ripple_filtered = bandpass_filter(raw, target_fs, low_freq, high_freq)
                ripple_power = np.sum(ripple_filtered ** 2, axis=0)
                max_ripple_shanks = np.argmax(ripple_power)

                combined_dict[probe][shank] = {
                    'spk.group': spk_group[probe][shank],
                    'max_ripple_power_channel': max_ripple_shanks,
                    #'max_theta_power_channel': max_theta_shanks[probe][shank]
                }

        #max_theta = find_max_value_and_index(theta_power)
        #combined_dict['max_theta'] = max_theta
        output_dict = {}
        for probe in spk_group:
            output_dict[probe] = {}
            for shank in spk_group[probe]:
                ripple_power_channel = combined_dict[probe][shank]['max_ripple_power_channel']
                #theta_power_channel = combined_dict[probe][shank]['max_theta_power_channel']
                output_dict[probe][shank] = {
                    'channel_information': combined_dict[probe][shank],
                    'ripple_channel': shank_signals[probe][shank][:,ripple_power_channel],
                    'power_spectrum':  downsampled_power_spectrum[probe][shank],
                    'lfp': shank_signals[probe][shank],
                }
            #max_theta_shank = combined_dict['max_theta'][probe]['shank']
            #max_theta_index = combined_dict['max_theta'][probe]['index']
            #theta_power_shank =  shank_signals[probe][max_theta_shank]
            #output_dict[probe]['theta'] = theta_power_shank[:,max_theta_index]

        import pickle as pkl
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_ripple_power_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(output_directory , output_filename), 'wb') as file:
            pkl.dump(output_dict, file)
        del power_spectrum
        del downsampled_power_spectrum
        del output_dict
        del combined_dict
        del shank_signals



####### NREM DURATION ################

from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import*

for rat_index in range(0,4):
    low_freq = low_ripple_freq[rat_index]
    high_freq = high_ripple_freq[rat_index]
    print('Extraction Ripple Bands from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        eef_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.eeg'
        xml_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.xml'
        eeg_file_directory = os.path.join(rat_directory, session_directory,eef_file_name)
        xml_file_directory = os.path.join(rat_directory, session_directory,xml_file_name)
        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        session_info = convert_session_mat_to_dict(session_information_directory)
        params = get_recorging_parameters(xml_file_directory)

        #### start loading
        lfp_sr = params['lfpSamplingRate']
        ### get NREM period
        NREM_durections = session_info['epochs']['NREMEpoch'][1,:] - session_info['epochs']['NREMEpoch'][0,:]
        NREM_max_duration = np.max(NREM_durections)
        NREM_max_duration_index = np.argmax(NREM_durections)

        start_resting = int(session_info['epochs']['NREMEpoch'][0,NREM_max_duration_index]  )
        end_resting = int(session_info['epochs']['NREMEpoch'][1,NREM_max_duration_index])
        print('RAT ' + rat_names[rat_index] + str((end_resting - start_resting)/60))




#####################PLOTTING ###################

import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === Setup ===
figures_directory = os.path.join(base_directory, 'figures')
os.makedirs(figures_directory, exist_ok=True)

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def plot_and_save_lfp(signal, fs, duration, title, save_path):
    n_samples = int(fs * duration)
    signal = signal[:n_samples, :]

    plt.figure(figsize=(10, 6))
    for i in range(signal.shape[1]):
        plt.plot(np.arange(n_samples) / fs, signal[:, i] + i * 200, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Channels (bottom = deep)')
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# === Get mouse/session names ===
mouse_name = rat_names[rat_index]
session_name = rat_sessions[mouse_name][session_index]
fs = target_fs
duration = 1.0  # seconds of LFP to plot

for probe in shank_signals:
    for shank in shank_signals[probe]:
        raw_lfp = shank_signals[probe][shank]  # shape: (time, channels)
        raw_lfp = raw_lfp[:, ::-1]  # Flip vertically: deep at bottom

        base_name = f"{mouse_name}_{session_name}_{probe}_shank{shank}"
        raw_fig_path = os.path.join(figures_directory, f"{base_name}_raw.png")
        ripple_fig_path = os.path.join(figures_directory, f"{base_name}_ripple.png")

        # Raw signal
        plot_and_save_lfp(raw_lfp, fs, duration,
                          title=f"Raw LFP - {base_name}",
                          save_path=raw_fig_path)

        # Ripple band
        ripple_lfp = bandpass_filter(raw_lfp, fs, 100, 250)
        plot_and_save_lfp(ripple_lfp, fs, duration,
                          title=f"Ripple Band (100–250 Hz) - {base_name}",
                          save_path=ripple_fig_path)


######################### PLOTTING RIPPLES #############################################


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, welch

# === Filtering functions ===
def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def lowpass_filter(data, fs, cutoff=10, order=4):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    return filtfilt(b, a, data, axis=0)

def normalize_channels(signal):
    return (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

# === Parameters ===
window_ms = 50
example_count = 100
spacing = 3
window_samples = int(fs * window_ms / 1000)
low_freq = 100
high_freq = 250

# === Create output folder ===
ripple_dir = os.path.join(figures_directory, 'ripple_examples')
os.makedirs(ripple_dir, exist_ok=True)

# === Loop through probes and shanks ===
for probe in shank_signals:
    for shank in shank_signals[probe]:
        raw = shank_signals[probe][shank]
        #reorder = [0,9,1,8,2,7,3,6,4,5]
        #raw = raw[:,reorder]
        ripple_filtered = bandpass_filter(raw, fs, low_freq, high_freq)
        ripple_power = np.sum(ripple_filtered**2, axis=0)
        max_chan = np.argmax(ripple_power)

        # Compute power spectra and AUC per channel
        psd_auc = []
        psd_all = []
        f_all = []
        nperseg = 1024 if raw.shape[0] > 1024 else raw.shape[0]
        for ch in range(raw.shape[1]):
            f, pxx = welch(raw[:, ch], fs=fs, nperseg=nperseg)
            f_all.append(f)
            normalized_power = pxx / np.max(pxx)
            psd_all.append(normalized_power)
            idx = (f >= low_freq) & (f <= high_freq)
            psd_auc.append(np.trapz(normalized_power[idx], f[idx]))
        psd_all = np.array(psd_all)  # shape: (channels, freqs)
        f = f_all[0]
        max_chan_psd = np.argmax(psd_auc)

        ripple_envelope = np.abs(hilbert(ripple_filtered[:, max_chan]))
        peak_times = np.argsort(ripple_envelope)[-example_count * 10:]
        peak_times = [pt for pt in peak_times if pt > window_samples and pt < len(raw) - window_samples]

        used = []
        examples_saved = 0

        for peak in peak_times:
            if any(abs(peak - u) < window_samples for u in used):
                continue
            used.append(peak)

            start = peak - window_samples
            end = peak + window_samples

            raw_window = raw[start:end]
            ripple_window = ripple_filtered[start:end]
            low_window = lowpass_filter(raw, fs)[start:end]

            # Normalize for plotting
            raw_window_norm = normalize_channels(raw_window)
            ripple_window_norm = normalize_channels(ripple_window)
            low_window_norm = normalize_channels(low_window)

            time_axis = np.linspace(-window_ms, window_ms, 2 * window_samples)

            # === Plotting ===
            fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharex=False)

            def plot_panel(ax, signal, title):
                for i in range(signal.shape[1]):
                    offset = i * spacing
                    color = 'black' if i == max_chan else 'dimgray'
                    color = 'gray' if i == max_chan else 'dimgray'
                    lw = 2.0 if i == max_chan else 0.8
                    lw = 2.0 if i == max_chan_psd else 0.8
                    ax.plot(time_axis, signal[:, i] + offset, color=color, linewidth=lw)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Channel")
                ax.set_yticks(np.arange(signal.shape[1]) * spacing)
                ax.set_yticklabels(np.arange(signal.shape[1]))
                ax.set_xlim(-window_ms, window_ms)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plot_panel(axs[0], raw_window_norm, "Raw LFP (normalized)")
            plot_panel(axs[1], ripple_window_norm, "Ripple Band (100–250 Hz)")
            plot_panel(axs[2], low_window_norm, "Low Frequencies (<10 Hz)")

            # === Power spectrum subplot (no normalization) ===
            ax = axs[3]
            offset_increment = 0.001
            nperseg = 1024 if raw.shape[0] > 1024 else raw.shape[0]

            # Plot the normalized power spectrum in ax2 and fill area under the curve
            for channel in range(raw.shape[1]):
                f, pxx = welch(raw[:, channel], fs=fs, nperseg=nperseg)
                idx = (f >= low_freq) & (f <= high_freq)
                normalized_power = pxx / np.max(pxx)
                offset_power = normalized_power + offset_increment * channel
                lw = 2 if channel == max_chan else 1
                lw = 2 if channel == max_chan_psd else 1
                linestyle = '-' if channel == max_chan else '--'
                linestyle = ':' if channel == max_chan_psd else '--'
                color = 'black' if channel == max_chan else 'dimgray'
                color = 'gray' if channel == max_chan_psd else 'dimgray'
                alpha = 1.0 if channel == max_chan else 0.5
                alpha = 1.0 if channel == max_chan_psd else 0.5
                ax.plot(f, offset_power, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)
                # Highlight the area under the curve between 100 Hz and 250 Hz
                idx = (f >= low_freq) & (f <= high_freq)
                ax.fill_between(f[idx], offset_increment * channel, offset_power[idx], color=color,
                                 alpha=0.3)
                # Calculate and display the area under the curve
                area = np.sum(normalized_power[idx])
                ax.text(250, offset_power[idx][-1], f'{area:.2f}', fontsize=9, verticalalignment='bottom',
                         horizontalalignment='right')

            ax.set_xlim(0, 300)
            ax.set_ylim(0, 0.01)
            ax.set_title("Power Spectrum (Welch)", fontsize=10)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Channel")
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            # === Save figure ===
            folder = os.path.join(ripple_dir, f"{probe}_shank{shank}")
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"example_{examples_saved:03d}.png")
            fname = os.path.join(folder, f"example_{examples_saved:03d}.svg")

            plt.savefig(fname, dpi=300)
            plt.close()

            examples_saved += 1
            if examples_saved >= example_count:
                break

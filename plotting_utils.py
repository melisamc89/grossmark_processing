from unittest import signals

import matplotlib.pyplot as plt
import numpy as np


def plot_probe_signals(shank_signals):
    # Determine the number of shanks per probe to set up the subplots
    num_shanks_probe1 = len(shank_signals['Probe1'])
    num_shanks_probe2 = len(shank_signals['Probe2'])

    # Create figure with two columns and rows equal to the maximum number of shanks in any probe
    fig, axes = plt.subplots(max(num_shanks_probe1, num_shanks_probe2), 2, figsize=(15, 20), sharex='col')

    # Ensure axes is a 2D array
    if num_shanks_probe1 == 1 or num_shanks_probe2 == 1:
        axes = np.expand_dims(axes, axis=0)
    offset = 2000
    # Plotting for Probe1
    for i, (shank_key, signals) in enumerate(shank_signals['Probe1'].items()):
        color_cycle = plt.cm.viridis(np.linspace(0, 1,signals.shape[1]))  # Color map for visibility
        ax = axes[i, 0]
        ax.set_title(f'{shank_key} of Probe1')
        for j in range(signals.shape[1]):
            # Plot each channel with an offset
            ax.plot(signals[:, j] + j * offset, color=color_cycle[j], alpha = 0.5) #  # Offset each channel for visibility
        ax.legend()

    # Plotting for Probe2
    for i, (shank_key, signals) in enumerate(shank_signals['Probe2'].items()):
        ax = axes[i, 1]
        color_cycle = plt.cm.viridis(np.linspace(0, 1,signals.shape[1]))  # Color map for visibility
        ax.set_title(f'{shank_key} of Probe2')
        for j in range(signals.shape[1]):
            ax.plot(signals[:, j] + j * offset, color=color_cycle[j], alpha = 0.5) # Offset each channel for visibility
        ax.legend()

    # Set common labels and title
    for ax in axes[:, 0]:
        ax.set_ylabel('Amplitude + Offset')
    for ax in axes[-1, :]:
        ax.set_xlabel('Sample Number')
    fig.suptitle('Signals by Shank and Probe')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
    plt.show()

def plot_power_spectra(probe_spectra):
    # Determine the number of shanks for the subplot grid
    num_shanks_probe1 = len(probe_spectra['Probe1'])
    num_shanks_probe2 = len(probe_spectra['Probe2'])
    # Create figure with subplots
    fig, axes = plt.subplots(max(num_shanks_probe1, num_shanks_probe2), 2, figsize=(10, 15), sharex='col')
    # Ensure axes is 2D if only one row
    if max(num_shanks_probe1, num_shanks_probe2) == 1:
        axes = np.expand_dims(axes, axis=0)
    # Define an offset increment
    offset_increment = 0.001  # Adjust based on your data for better visibility
    # Plotting for Probe1
    for i, shank_key in enumerate(probe_spectra['Probe1'].keys()):
        ax = axes[i, 0] if num_shanks_probe1 > 1 else axes[0, 0]
        ax.set_title(f'{shank_key} of Probe1')
        freq = probe_spectra['Probe1'][shank_key]['frequency']
        power = probe_spectra['Probe1'][shank_key]['power']
        offset = 0
        for j in range(power.shape[1]):
            p = power[:, j] / np.max(power[:, j])
            ax.plot(freq, p + offset, label=f'{shank_key}', alpha = 0.5)
            offset += offset_increment

    # Plotting for Probe2
    for i, shank_key  in enumerate(probe_spectra['Probe2'].keys()):
        ax = axes[i, 1] if num_shanks_probe2 > 1 else axes[0, 1]
        ax.set_title(f'{shank_key} of Probe2')
        offset = 0
        freq = probe_spectra['Probe2'][shank_key]['frequency']
        power = probe_spectra['Probe2'][shank_key]['power']
        for j in range(power.shape[1]):
            p = power[:, j] / np.max(power[:, j])
            ax.plot(freq, p + offset, label=f'{shank_key}', alpha = 0.5)
            offset += offset_increment
    # Set common labels and title
    for ax in axes[:, 0]:
        ax.set_ylabel('Power + Offset')
        ax.set_ylim([0,offset_increment*10])
    for ax in axes[:, 1]:
        ax.set_ylim([0,offset_increment * 10])
    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)')

    fig.suptitle('Power Spectra by Shank and Probe')
    #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
    plt.show()
    return fig


def plot_single_spike_waveform(waveforms, spike_index=0, color_map='viridis'):
    """
    Plots waveforms for a single spike from multiple channels with vertical offsets and color coding.

    Args:
    waveforms (numpy.ndarray): 3D array with shape (num_spikes, num_channels, samples_per_spike)
    spike_index (int): Index of the spike to plot.
    color_map (str): Name of the matplotlib colormap to use for different channels.
    """
    num_channels = waveforms.shape[2]
    samples_per_spike = waveforms.shape[1]

    # Setup color map
    cmap = plt.get_cmap(color_map)
    colors = cmap(np.linspace(0, 1, num_channels))

    # Calculate offsets to separate the waveforms vertically
    offsets = 100

    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot each channel's waveform with an offset
    for channel in range(num_channels):
        offset_waveform = waveforms[spike_index, :, channel] + offsets*channel
        ax.plot(offset_waveform, label=f'Channel {channel + 1}', color=colors[channel])

    # Add labels and title
    ax.set_title('Single Spike Waveforms Across Channels')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude + Offset')
    ax.legend(loc='upper right')

    # Add grid for better readability
    ax.grid(True)

    plt.show()


def plot_waveform_and_power(waveforms, power_spectrum, limit_channel,max_peak, text, low_freq,high_freq,neuron_type,color_map='viridis'):
    num_channels = waveforms.shape[1]
    samples_per_spike = waveforms.shape[0]
    n_freqs = power_spectrum['power'].shape[0]

    # Setup color map
    cmap = plt.get_cmap(color_map)
    colors = cmap(np.linspace(0, 1, num_channels))

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Calculate offsets to separate the waveforms and power spectra vertically
    offset_increment = 0.001  # Adjust based on your data for better visibility

    # Plot each channel's waveform with an offset in ax1
    for channel in range(waveforms.shape[1]):
        offset_waveform = waveforms[:, channel] + offset_increment * channel
        if channel == limit_channel:
            ax1.plot(offset_waveform, label=f'LIMITChannel {channel + 1}', color=colors[channel], linestyle='-', linewidth=2,
                     alpha=1)
        else:
            ax1.plot(offset_waveform, label=f'Channel {channel + 1}', color=colors[channel], linestyle='--', alpha=0.5)
        if channel == max_peak:
            ax1.plot(offset_waveform, label=f'MAXChannel {channel + 1}', color = 'Red', linestyle='--', linewidth=2,
                     alpha=1)

    colors = cmap(np.linspace(0, 1, power_spectrum['power'].shape[1]))

        # Plot the normalized power spectrum in ax2 and fill area under the curve
    for channel in range(power_spectrum['power'].shape[1]):
        frequencies = power_spectrum['frequency']
        power_values = power_spectrum['power'][:, channel]
        normalized_power = power_values / np.max(power_values)
        offset_power = normalized_power + offset_increment * channel
        color = colors[channel]
        if channel == limit_channel:
            ax2.plot(frequencies, offset_power, color=colors[channel], linestyle='-', linewidth=2, alpha=1)
        else:
            ax2.plot(frequencies, offset_power,
                     color=colors[channel], linestyle='--', alpha=0.5)

        # Highlight the area under the curve between 100 Hz and 250 Hz
        idx = (frequencies >= low_freq) & (frequencies <= high_freq)
        ax2.fill_between(frequencies[idx], offset_increment * channel, offset_power[idx], color=color, alpha=0.3)

        # Calculate and display the area under the curve
        area = np.sum(normalized_power[idx])
        ax2.text(250, offset_power[idx][-1], f'{area:.2f}', fontsize=9, verticalalignment='bottom',
                 horizontalalignment='right')

    # Setting labels and titles
    ax1.set_title('CLASSFICATION:' + text + ' TYPE: ' + neuron_type, fontsize=20)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude + Offset')
    ax1.legend(loc='upper right')
    ax1.grid(False)

    ax2.set_title('Normalized Power Spectrum Across Channels')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Power + Offset')
    #ax2.set_yscale('log')  # Assuming frequency needs a log scale for better visualization
    ax2.grid(False)
    ax2.set_ylim([0,10*offset_increment])

    plt.tight_layout()
    plt.show()
    return fig


    # Setting labels and titles
    ax1.set_title('Single Spike Waveforms Across Channels CLASS: ' + text, fontsize=20)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude + Offset')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.set_title('Normalized Power Spectrum Across Channels')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Power + Offset')
    ax2.grid(True)
    ax2.set_ylim([0, 10 * offset_increment])

    plt.tight_layout()
    plt.show()
    return fig

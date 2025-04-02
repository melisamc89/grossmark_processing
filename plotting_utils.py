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
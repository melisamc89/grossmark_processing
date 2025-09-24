

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
    'perc_neigh': 100,
    'num_shuffles': 0,
    'verbose': False
}

params2 = {
    'n_bins': 3,
    'discrete_label': True,
    'continuity_kernel': None,
    'n_neighbors': 1,
    'num_shuffles': 0,
    'verbose': False
}

si_neigh = 100
si_beh_params = {}
for beh in ['pos', 'speed', 'trial_id_mat','time','(pos,dir)']:
    si_beh_params[beh] = copy.deepcopy(params1)
for beh in ['dir']:
    si_beh_params[beh] = copy.deepcopy(params2)

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


neural_data_dir = files_directory

rats = [0]
sessions = [[0],[0]]
speed_lim = 0.05 #(m/s)
for rat_index in rats:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1','Probe2']:
            si_dict= dict()
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'neural_data_2.pkl'
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
                '(pos,dir)': position*direction,
                'speed': speed,
                'trial_id_mat': trial_id,
                'dir': direction,
                'time': time
            }

            spikes_matrix = stimes['spikes_matrix']
            #spikes_matrix = preprocess_spikes(spikes_matrix)
            spikes_matrix = spikes_matrix[valid_mov,:]
            layerID = stimes['LayerID']
            typeID = stimes['TypeID']
            # Find indices where the value is 'DEEP'
            deep_index = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'PYR'])
            # Find indices where the value is 'SUPERFICIAL'
            superficial_index = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'PYR'])
            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])

            kernels = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
            for beh_name, beh_val in beh_variables.items():
                if 'trial_id_mat' in beh_name:
                    si_beh_params[beh_name]['min_label'] = np.min(beh_val)
                    si_beh_params[beh_name]['max_label'] = np.max(beh_val)
                # si_beh_params[beh_name]['n_neighbors'] = int(signal.shape[0]* si_beh_params[beh_name]['perc_neigh']/100)
                si_beh_params[beh_name]['n_neighbors'] = si_neigh
                si_dict[beh_name] = dict()

                for index, filter_size in enumerate(kernels):
                    #si_beh_params[beh_name]['n_neighbors'] = si_neigh + 5 * filter_size
                    data = spikes_to_rates(spikes_matrix.T, kernel_width=filter_size)
                    deep_spikes = data[:,deep_index]
                    sup_spikes =  data[:,superficial_index]
                    data=data[:,pyr_index]
                    print(data.shape)

                    si, process_info, overlap_mat, _ = compute_structure_index(data,beh_val,
                                                                               **si_beh_params[beh_name])

                    si_deep, process_info, overlap_mat, _ = compute_structure_index(deep_spikes,beh_val,
                                                                               **si_beh_params[beh_name])
                    si_sup, process_info, overlap_mat, _ = compute_structure_index(sup_spikes,beh_val,
                                                                               **si_beh_params[beh_name])

                    si_dict[beh_name][str(filter_size)] = {
                        'si': copy.deepcopy(si),
                        'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                        'si_deep': copy.deepcopy(si_deep),
                        'si_sup': copy.deepcopy(si_sup),
                    }

                    print(f" {beh_name}_{filter_size}={si:.4f} |", end='', sep='', flush='True')
                    print()

            data_output_directory = '/home/melma31/Documents/time_project/SI_filters'
            # Define the filename where the dictionary will be stored
            output_filename = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] +'_'+ probe + '_si_100_2-60_2.pkl'
            # Open the file for writing in binary mode and dump the dictionary
            with open(os.path.join(data_output_directory, output_filename), 'wb') as file:
                pkl.dump(si_dict, file)


###############################################
#
#       PLOTTING SI VALUES VS FILTERS       #
#
#############################################


data_output_directory = '/home/melma31/Documents/time_project/SI_filters'
rats = [0]
sessions = [[0],[0],[0,2],[0]]
probes_used = [[['Probe1','Probe2']],[['Probe1']],[['Probe1','Probe2'],['Probe1','Probe2']],[['Probe1','Probe2']]]
import os
import pickle as pkl
import pandas as pd
# Initialize list to collect rows
df_rows = []
# Lop through the rat data structure
for rat_index in rats:
    rat_name = rat_names[rat_index]
    for count_session, session_index in enumerate(sessions[rat_index]):
        for probe in probes_used[rat_index][count_session]:
            # Construct filename (assuming output_filename uses rat_name and session)
            output_filename = rat_name + '_' + rat_sessions[rat_name][session_index] + '_' + probe + '_si_variable_nei_2-60_assimetry.pkl'
            file_path = os.path.join(data_output_directory, output_filename)
            # Load the .pkl dictionary
            with open(file_path, 'rb') as file:
                si_dict = pkl.load(file)
            # Access the data for this session and probe
            # Loop over behavioral labels and filter sizes
            for beh_label, filter_data in si_dict.items():
                for filter_size, data in filter_data.items():
                    row = {
                        'rat': rat_name,
                        'session': session_index,
                        'probe': probe,
                        'filter': int(filter_size),
                        'behavioral_label': beh_label,
                        'si': data['si'],
                        'si_deep': data['si_deep'],
                        'si_sup': data['si_sup'],
                        # Optional: include behavior params too
                        # 'beh_params': data['beh_params']
                    }
                    df_rows.append(row)
# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: Sort for readability
df.sort_values(by=['rat', 'session', 'probe', 'behavioral_label', 'filter'], inplace=True)
# Preview
print(df.head())



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Set plotting style
sns.set(style="whitegrid")

# Melt your DataFrame to long format to handle si, si_deep, si_sup as "Area"
df_long = df.melt(id_vars=['rat', 'session', 'probe', 'filter', 'behavioral_label'],
                  value_vars=['si', 'si_deep', 'si_sup'],
                  var_name='Area', value_name='SI')
# Map custom colors and area names

area_palette = {
    'si': '#bbbcc0ff',
    'si_deep': '#cc9900',
    'si_sup': '#9900ff',
}

area_labels = {
    'si': 'Overall',
    'si_deep': 'Deep',
    'si_sup': 'Superficial'
}

# Create subplots
behavior_labels = df_long['behavioral_label'].unique()
fig, axes = plt.subplots(len(behavior_labels), 1, figsize=(8, 2 * len(behavior_labels)), sharex=True)

if len(behavior_labels) == 1:
    axes = [axes]

for i, beh in enumerate(behavior_labels):
    ax = axes[i]

    # Subset for this behavior
    sub_df = df_long[df_long['behavioral_label'] == beh]

    # Lineplot with shaded error band (std)
    sns.lineplot(data=sub_df, x='filter', y='SI', hue='Area',
                 ax=ax, errorbar='sd', palette=area_palette, marker='o')

    # Formatting
    ax.set_title(f'Structure Index vs Filter Size — {beh}')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_ylim(0, 1)
    ax.grid(False)

    if i == len(behavior_labels) - 1:
        ax.set_xlabel('Filter Size')
    else:
        ax.set_xlabel('')

    # Legend with friendly names
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [area_labels.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title='Area')

# Layout
#fig.suptitle('Structure Index vs Filter Size (Mean ± SD)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
fig.savefig(os.path.join(figures_directory, 'Si_rats_a_si_100_2-60.png'))




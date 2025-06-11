

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

rats = [0,1]
sessions = [[0],[0]]
speed_lim = 0.05 #(m/s)
for rat_index in rats:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1','Probe2']:
            si_dict= dict()
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

            kernels = [5, 10, 15, 20, 25, 30]
            umap_model = umap.UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)
            spikes_matrix_list = [spikes_matrix,spikes_matrix[:,deep_index],spikes_matrix[:,superficial_index]]
            figure_title = ['PYRAMIDAL CELLS: ' + str(spikes_matrix.shape[1]),
                          'DEEP PYRAMIDAL CELLS: ' + str(deep_index.shape[0]),
                          'SUPERFICIAL  PYRAMIDAL CELLS: ' + str(superficial_index.shape[0]), ]
            figure_data = ['pyr','deep_pyr','sup_pyr']
            for fig_idx, spikes_ in enumerate(spikes_matrix_list):
                row = 3
                col = 6
                fig = plt.figure(figsize=(15, 9))
                for index, filter_size in enumerate(kernels):
                    data = spikes_to_rates(spikes_.T, kernel_width=filter_size)
                    umap_model.fit(data)
                    umap_emb = umap_model.transform(data)

                    for beh_name, beh_val in beh_variables.items():
                        # --- Position ---
                        ax = fig.add_subplot(row, col, 1 + index, projection='3d')
                        ax.set_title('Position')
                        ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=position, s=1, alpha=0.5,
                                   cmap='magma')
                        ax.view_init(elev=30, azim=30)
                        ax.grid(False)

                        # --- Direction ---
                        ax = fig.add_subplot(row, col, 7 + index, projection='3d')
                        ax.set_title('Direction')
                        ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=direction, s=1, alpha=0.5,
                                   cmap='Blues')
                        ax.view_init(elev=30, azim=30)
                        ax.grid(False)

                        # --- Time ---
                        ax = fig.add_subplot(row, col, 13 + index, projection='3d')
                        ax.set_title('Time')
                        ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=np.arange(0, umap_emb.shape[0]),
                                   s=1,
                                   alpha=0.5, cmap='Greens')
                        ax.view_init(elev=30, azim=30)
                        ax.grid(False)

                fig.suptitle(figure_title[fig_idx], fontsize=16)

                figure_name = rat_names[rat_index] + '_' + str(
                    rat_sessions[rat_names[rat_index]][session_index]) + '_' + probe + 'umap_' + figure_data[
                                                                                                 fig_idx] + '_filters.png'
                fig.savefig(os.path.join(figures_directory, 'filters_examples', figure_name))


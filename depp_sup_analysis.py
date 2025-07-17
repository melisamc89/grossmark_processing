

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


import matplotlib.pyplot as plt

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

neural_data_dir = files_directory

rat_index = [0,1,2,3]
sessions = [[0],[0],[2],[1]]
speed_lim = 0.05

for rat_index in [0]:
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

            #file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'POST_neural_data.pkl'
            #spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode
            #with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
            #    stimes_NREM = pkl.load(file)

            k = 10
            spikes_matrix = stimes['spikes_matrix']
            spikes_matrix = spikes_matrix[valid_mov,:]
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
            #spikes_matrixNREM = stimes_NREM['spikes_matrix']
            #data_NREM = spikes_to_rates(spikes_matrixNREM.T, kernel_width=k)

            layerID = stimes['LayerID']
            typeID = stimes['TypeID']
            # Find indices where the value is 'DEEP'
            deep_index = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'PYR'])
            # Find indices where the value is 'SUPERFICIAL'
            superficial_index = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'PYR'])

            deep_spikes = data[:,deep_index]
            sup_spikes =  data[:,superficial_index]

            #deep_spikes_NREM = data_NREM[:,deep_index]
            #sup_spikes_NREM =  data_NREM[:,superficial_index]

            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])
            # Find indices where the value is 'SUPERFICIAL'
            data=data[:,pyr_index]
            #data_NREM = data_NREM[:,pyr_index]
            #deep_index_int = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
             #                      value == 'DEEP' and ntype == 'INT'])
            # Find indices where the value is 'SUPERFICIAL'
            #superficial_index_int = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
            #                              value == 'SUPERFICIAL' and ntype == 'INT'])

            #deep_spikes_int = data[:,deep_index_int]
            #sup_spikes_int =  data[:,superficial_index_int]


            umap_model = umap.UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)

            umap_model.fit(data)
            umap_emb_all = umap_model.transform(data)
            #umap_emb_all_NREM = umap_model.transform(data_NREM)

            umap_model.fit(deep_spikes)
            umap_emb_deep = umap_model.transform(deep_spikes)
            #umap_emb_deep_NREM = umap_model.transform(deep_spikes_NREM)

            umap_model.fit(sup_spikes)
            umap_emb_sup = umap_model.transform(sup_spikes)
            #umap_emb_sup_NREM = umap_model.transform(sup_spikes_NREM)

            #umap_emb_ = [umap_emb_all,umap_emb_all_NREM,umap_emb_deep,umap_emb_deep_NREM , umap_emb_sup,umap_emb_sup_NREM]
            umap_emb_ = [umap_emb_all,umap_emb_deep,umap_emb_sup]

            umap_title = ['ALL: ' + str(data.shape[1]),
                          'DEEP: ' + str(deep_spikes.shape[1]) ,
                          'SUPERFICIAL: ' + str(sup_spikes.shape[1]),]

            row = 3
            col = 2
            fig = plt.figure(figsize=(8, 12))
            for index ,umap_emb in enumerate(umap_emb_):
                #if index != 1 and index!=3 and index!=5:
                ax = fig.add_subplot(row, col, 1 + 2*index, projection='3d')
                # ax = fig.add_subplot(row, col, index + 1)
                ax.set_title('Position ' + umap_title[index])
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=position, s=1, alpha=0.5, cmap='magma')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = labels[:-1], s= 1, alpha = 0.5, cmap = 'magma')
                ax.grid(False)

                #ax = fig.add_subplot(row, col, 2 + 5*index, projection='3d')
                # ax = fig.add_subplot(row, col, index + 1)
                #ax.set_title('Direction ' + umap_title[index])
                #ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=direction, s=1, alpha=0.5, cmap='Blues')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = labels[:-1], s= 1, alpha = 0.5, cmap = 'magma')
                #ax.grid(False)

                #ax = fig.add_subplot(row, col, 3 + 5 * index, projection='3d')
                # ax = fig.add_subplot(row, col, index + len(kernels) + 1 )
                #ax.set_title('Speed ' + umap_title[index])
                #ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=speed, s=1, alpha=0.5, cmap='Reds')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = speed, s= 1, alpha = 0.5, cmap = 'Reds')
                #ax.grid(False)

                ax = fig.add_subplot(row, col, 2 + 2*index, projection='3d')
                # ax = fig.add_subplot(row, col,index + len(kernels)*2 + 1)
                ax.set_title('Time' + umap_title[index])
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=np.arange(0, umap_emb.shape[0]), s=1,
                               alpha=0.5, cmap='YlGn_r')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = np.arange(0,umap_emb.shape[0]), s = 1, alpha = 0.5, cmap = 'Greens')
                ax.grid(False)

                #ax = fig.add_subplot(row, col, 5 + 5*index, projection='3d')
                # ax = fig.add_subplot(row, col,index + len(kernels) * 3 +1 )
                #ax.set_title('Trial ID '+ umap_title[index])
                #ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=trial_id, s=1, alpha=0.5, cmap='viridis')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = trial_id[:-1], s = 1, alpha = 0.5, cmap = 'viridis')
                #ax.grid(False)

                fig.tight_layout()

            # Define the filename where the dictionary will be stored
            figure_name = rat_names[rat_index] + '_' + str(rat_sessions[rat_names[rat_index]][session_index]) +'_' +probe + 'umap_deep_sup_filter_'+str(k)+'.png'
            fig.savefig(os.path.join(figures_directory, figure_name))




########################
########################
#######################


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

rat_index = [0]
sessions = [[0]]
speed_lim = 0.05

for rat_index in [0]:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1','Probe2']:
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'POST_neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode

            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)

            k = 10
            spikes_matrix = stimes['spikes_matrix']
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
            layerID = stimes['LayerID']
            typeID = stimes['TypeID']

            time = np.arange(0,spikes_matrix.shape[0])
            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])
            data=data[:,pyr_index]
            X = data.copy()
            # Step 1: Define weight matrix W for Hopfield-like energy
            # Use Hebbian rule based on a few "pattern" states (e.g., selected timepoints)
            # Assume data is your original matrix of shape (time, neurons)
            time_points = data.shape[0]
            n_sample = int(0.1 * time_points)  # 10% of temporal points
            # Randomly choose indices without replacement
            selected_indices = np.random.choice(time_points, size=n_sample, replace=False)
            pattern_indices = selected_indices
            patterns = X[pattern_indices]
            W = np.sum([np.outer(p, p) for p in patterns], axis=0)
            # Ensure W is symmetric and zero diagonal (optional)
            np.fill_diagonal(W, 0)
            # Step 2: Compute energy over time
            # E(x_t) = -0.5 * x_t^T W x_t
            energy_hopfield = -0.5 * np.einsum('ij,ij->i', X @ W, X)
            # Step 3: Plot energy over time
            plt.figure(figsize=(10, 4))
            plt.plot(energy_hopfield, color='crimson', label='Hopfield-Like Energy')
            plt.xlabel('Time')
            plt.ylabel('Energy')
            plt.title('Hopfield-Like Quadratic Energy over Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()



            # Smooth the Hopfield energy signal
            from scipy.signal import find_peaks

            energy_hopfield_smooth = gaussian_filter1d(energy_hopfield, sigma=3)

            # Find local minima
            minima_indices_hopfield, _ = find_peaks(-energy_hopfield_smooth, distance=50)

            # Create label array for UMAP coloring: 1 = local minima, 0 = otherwise

            colors_hopfield = np.zeros_like(energy_hopfield, dtype=int)
            colors_hopfield[minima_indices_hopfield] = 1

            # Plot energy over time and UMAP highlighting local minima
            fig = plt.figure(figsize=(12, 10))

            # Top: Energy over time with minima marked
            ax1 = fig.add_subplot(211)
            ax1.plot(energy_hopfield, color='darkorange', label='Hopfield Energy')
            ax1.plot(minima_indices_hopfield, energy_hopfield[minima_indices_hopfield],
                     'bo', label='Local Minima')
            ax1.set_title('Hopfield-Like Energy over Time with Minima')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Energy')
            ax1.legend()
            ax1.grid(True)

            # Bottom: UMAP embedding colored by local minima
            ax2 = fig.add_subplot(212, projection='3d')
            # sc = ax2.scatter(X_umap[:, 1], X_umap[:, 2], X_umap[:, 0],
            #                 c=dir_color, s=10, alpha = 0.05)
            ax2.set_title('UMAP Embedding: Local Minima Highlighted')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.set_zlabel('UMAP 3')
            fig.colorbar(sc, ax=ax2, ticks=[0, 1], label='0 = Not Minima, 1 = Minima', shrink=0.6, pad=0.1)
            ax2.scatter(X_umap[minima_indices_hopfield, 1], X_umap[minima_indices_hopfield, 2],
                        X_umap[minima_indices_hopfield, 0],
                        c=beh['time'][minima_indices_hopfield], cmap='YlGn', s=50)
            plt.tight_layout()
            plt.show()



            umap_model = umap.UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)

            umap_model.fit(data)
            umap_emb_all = umap_model.transform(data)
            #umap_emb_all_NREM = umap_model.transform(data_NREM)

            umap_emb_ = [umap_emb_all]
            umap_title = ['ALL: ' + str(data.shape[1])]

            row = 1
            col = 1
            fig = plt.figure(figsize=(15, 16))
            for index ,umap_emb in enumerate(umap_emb_):
                #if index != 1 and index!=3 and index!=5:
                ax = fig.add_subplot(row, col, 1 + 5*index, projection='3d')
                # ax = fig.add_subplot(row, col, index + 1)
                ax.set_title('Position ' + umap_title[index])
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=time, s=1, alpha=0.5, cmap='magma')
                # ax.scatter(umap_emb[:,0],umap_emb[:,1], c = labels[:-1], s= 1, alpha = 0.5, cmap = 'magma')
                ax.grid(False)

            # Define the filename where the dictionary will be stored
            figure_name = rat_names[rat_index] + '_' + str(rat_sessions[rat_names[rat_index]][session_index]) +'_' +probe + 'umap_all_filter_'+str(k)+'.png'
            fig.savefig(os.path.join(figures_directory, figure_name))



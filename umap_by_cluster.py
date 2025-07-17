

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

from src.config import *
# Setup
cluster_directory = os.path.join(base_directory, 'clusters')

files_names = ['Achilles_mi_transferred_cluster_3_dict.pkl',
               'Calb_mi_transferred_cluster_3_dict.pkl',
               'Thy_mi_transferred_cluster_3_dict.pkl']
mouse_names = ['Achilles', 'Calb', 'Thy']


#files_names = ['Calb_mi_transferred_cluster_3_dict.pkl',
#               'Thy_mi_transferred_cluster_3_dict.pkl']
#mouse_names = ['Calb', 'Thy']


behavior_keys = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']

# Initialize storage
mouse_name_list, area_list, probe_list, typeID_list = [], [], [], []
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}
import pickle
# Load data
speed_limit = {'Achilles': 0.05, 'Calb': 6, 'Thy': 6}
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from scipy.ndimage import gaussian_filter1d
import umap

# Ensure your figures directory exists
os.makedirs(figures_directory, exist_ok=True)

for fname, mouse in zip(files_names, mouse_names):
    filepath = os.path.join(cluster_directory, fname)
    with open(filepath, 'rb') as f:
        mi_dict = pickle.load(f)

    for probe, probe_data in mi_dict.items():
        signal = probe_data['signal']
        clusterID = probe_data['clusterID']
        beh_variables = {
            'pos': probe_data['behaviour']['position'],
            '(pos,dir)': probe_data['behaviour']['(pos,dir)'],
            'speed': probe_data['behaviour']['speed'],
            'trial_id_mat': probe_data['behaviour']['trial_id'],
            'dir': probe_data['behaviour']['mov_dir'],
            'time': probe_data['behaviour']['time'],
        }

        valid_index = np.where(probe_data['behaviour']['speed'] > speed_limit[mouse])[0]
        clusters_names = np.unique(clusterID)

        # Apply Gaussian filter to full signal
        filtered_signal = gaussian_filter1d(signal, sigma=6, axis=0)

        # Compute UMAP for full signal in 3D
        umap_model = umap.UMAP(n_neighbors=120, n_components=3, random_state=42)
        embedding_full = umap_model.fit_transform(filtered_signal[valid_index])

        pos = beh_variables['pos'][valid_index]
        time = beh_variables['time'][valid_index]

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'UMAP 3D embeddings for {mouse} - Probe {probe}')

        # Row 1, Col 1: Full signal colored by position
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        p1 = ax1.scatter(embedding_full[:, 0], embedding_full[:, 1], embedding_full[:, 2],
                         c=pos, cmap='magma', s=1)
        ax1.set_title(f'Full Signal (N={signal.shape[1]}) - Pos')
        fig.colorbar(p1, ax=ax1, shrink=0.5)

        # Row 2, Col 1: Full signal colored by time
        ax2 = fig.add_subplot(2, 4, 5, projection='3d')
        p2 = ax2.scatter(embedding_full[:, 0], embedding_full[:, 1], embedding_full[:, 2],
                         c=time, cmap='YlGn_r', s=1)
        ax2.set_title(f'Full Signal (N={signal.shape[1]}) - Time')
        fig.colorbar(p2, ax=ax2, shrink=0.5)

        cluster_count = 0
        for cluster_idx in clusters_names:
            if cluster_idx == 10:
                continue
            if cluster_count >= 3:
                break

            cluster_mask = (clusterID == cluster_idx)
            cluster_signal = signal[:, cluster_mask]
            cluster_signal = cluster_signal[valid_index]

            n_neurons = cluster_signal.shape[1]
            if n_neurons < 2:
                print(f"Cluster {cluster_idx} too small, skipping.")
                continue

            # Apply Gaussian filter to cluster signal
            filtered_cluster_signal = gaussian_filter1d(cluster_signal, sigma=6, axis=0)

            # Compute UMAP for cluster signal in 3D
            umap_model = umap.UMAP(n_neighbors=120, n_components=3, random_state=42)
            embedding_cluster = umap_model.fit_transform(filtered_cluster_signal)

            # Row 1: Cluster colored by position
            ax_pos = fig.add_subplot(2, 4, cluster_count + 2, projection='3d')
            p3 = ax_pos.scatter(embedding_cluster[:, 0], embedding_cluster[:, 1], embedding_cluster[:, 2],
                                c=pos, cmap='magma', s=1)
            ax_pos.set_title(f'Cluster {cluster_idx} (N={n_neurons}) - Pos')
            fig.colorbar(p3, ax=ax_pos, shrink=0.5)

            # Row 2: Cluster colored by time
            ax_time = fig.add_subplot(2, 4, cluster_count + 6, projection='3d')
            p4 = ax_time.scatter(embedding_cluster[:, 0], embedding_cluster[:, 1], embedding_cluster[:, 2],
                                 c=time, cmap='YlGn_r', s=1)
            ax_time.set_title(f'Cluster {cluster_idx} (N={n_neurons}) - Time')
            fig.colorbar(p4, ax=ax_time, shrink=0.5)

            cluster_count += 1

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)

        probe_clean = str(probe).replace(" ", "_")
        save_name = f"{mouse}_{probe_clean}_umap_3D_clusters.png"
        save_path = os.path.join(figures_directory, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved 3D figure for {mouse} - {probe} at {save_path}")

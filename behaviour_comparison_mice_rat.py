

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


import matplotlib.pyplot as plt
# Initialize storage for all data
all_positions = []
all_speeds = []
all_times = []
# Load data once for all mice
for fname, mouse in zip(files_names, mouse_names):
    filepath = os.path.join(cluster_directory, fname)
    with open(filepath, 'rb') as f:
        mi_dict = pickle.load(f)
    # Take the first probe for this summary (or you can average across probes if needed)
    for probe_data in mi_dict.values():
        pos = probe_data['behaviour']['position']
        speed = probe_data['behaviour']['speed']
        time = probe_data['behaviour']['time']
        all_positions.append(pos)
        all_speeds.append(speed)
        all_times.append(time)
        break  # Only first probe per mouse for summary
# === PLOT: POSITION OVER TIME ===
fig1, axes1 = plt.subplots(len(mouse_names), 1, figsize=(12, 3 * len(mouse_names)))
if len(mouse_names) == 1:
    axes1 = [axes1]  # Make iterable if only one mouse

for idx, (mouse, pos, time) in enumerate(zip(mouse_names, all_positions, all_times)):
    if idx == 0:
        fs_vector = 40
    else:
        fs_vector = 20
    axes1[idx].plot(pos[0:fs_vector*10], color='tab:blue')
    axes1[idx].set_ylabel('Position')
    axes1[idx].set_title(f'{mouse} - Position over Time')
    axes1[idx].grid(False)

axes1[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig(os.path.join(figures_directory, 'Position_over_Time_all_mice.png'))
plt.show()
# === PLOT: SPEED OVER TIME ===
fig2, axes2 = plt.subplots(len(mouse_names), 1, figsize=(12, 3 * len(mouse_names)), sharex=True)
if len(mouse_names) == 1:
    axes2 = [axes2]  # Make iterable if only one mouse
for idx, (mouse, speed, time) in enumerate(zip(mouse_names, all_speeds, all_times)):
    if idx == 0:
        fs_vector = 40
    else:
        fs_vector = 20
    axes2[idx].plot(speed[0:fs_vector*10], color='tab:orange')
    axes2[idx].set_ylabel('Speed')
    axes2[idx].set_title(f'{mouse} - Speed over Time')
    axes2[idx].grid(False)
axes2[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig(os.path.join(figures_directory, 'Speed_over_Time_all_mice.png'))
plt.show()

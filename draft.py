import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import viridis

# Example data points in 3D space
points = umap_emb_all
#points = umap_emb_all[1000:9000,:]


# Example flow magnitudes between consecutive points
differences = np.diff(points, axis=0)
flow_intensities = np.linalg.norm(differences, axis=1)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
labels = position


N= points.shape[0]
#ax.scatter(points[:N,0], points[:N,1], points[:N,2], c=labels[:N], s=1, alpha=0.5, cmap='viridis')

# Normalize differences to get unit vectors
unit_vectors = differences#/ np.linalg.norm(differences, axis=1)[:, np.newaxis]
# Normalize these labels to use with the colormap
x = flow_intensities
norm = Normalize(vmin=labels.min(), vmax=labels.max())

color_mapper = viridis

# Adding arrows with color mapped from predefined labels
for i in range(len(flow_intensities)):
    color = color_mapper(norm(labels[i]))
    #color = color_mapper(norm([i]))
    ax.quiver(points[i,0], points[i,1], points[i,2],
              unit_vectors[i,0], unit_vectors[i,1], unit_vectors[i,2],
              length=flow_intensities[i],
              #arrow_length_ratio=0.5,
              color=color)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)

# Add a color bar to show label mapping
sm = plt.cm.ScalarMappable(cmap=color_mapper, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Position')
# Show the plot
plt.show()


####RUN FROM HERE

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

###############################################
#
#       PLOTTING SI VALUES VS FILTERS       #
#
#############################################


data_output_directory = '/home/melma31/Documents/time_project/SI_filters'
rats = [0,1,2,3]
sessions = [[0],[0],[0,2],[0]]
probes_used = [[['Probe1','Probe2']],[['Probe1']],[['Probe1','Probe2'],['Probe1','Probe2']],[['Probe1','Probe2']]]

rats = [0]
sessions = [[0]]
probes_used = [[['Probe1','Probe2']],[['Probe1']]]

#rats = []
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
            output_filename = rat_name + '_' + rat_sessions[rat_name][session_index] + '_' + probe + '_si_variable_nei_2-60.pkl'
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

#### ADD TERESA'S
mouse_names = ['Calb','Thy']
for mouse in mouse_names:
    output_filename = f"{mouse}_mouse_si_variable_nei_2-60.pkl"
    file_path = os.path.join(data_output_directory, output_filename)
    # Load the .pkl dictionary
    with open(file_path, 'rb') as file:
        si_dict = pkl.load(file)
    # Access the data for this session and probe
    # Loop over behavioral labels and filter sizes
    for beh_label, filter_data in si_dict.items():
        for filter_size, data in filter_data.items():
            row = {
                'rat': mouse,
                'session': 1,
                'probe': 1,
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

# Convert filter size to time in seconds
sampling_rate = 40  # Hz
df_long = df.melt(id_vars=['rat', 'session', 'probe', 'filter', 'behavioral_label'],
                  value_vars=['si', 'si_deep', 'si_sup'],
                  var_name='Area', value_name='SI')
df_long['filter_time'] = df_long['filter'] / sampling_rate
# Define allowed time values (in seconds)
allowed_times = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Filter DataFrame to only include allowed times
df_long = df_long[df_long['filter_time'] <= 1.5]


# Map custom colors and area names
area_palette = {
    'si': 'black',
    'si_deep': 'gold',
    'si_sup': 'purple'
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
    sns.lineplot(data=sub_df, x='filter_time', y='SI', hue='Area',
                 ax=ax, errorbar='sd', palette=area_palette, marker='o')

    # Formatting
    ax.set_title(f'Structure Index vs Filter Time — {beh}')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_ylim(0, 1)
    ax.grid(False)

    #if i == len(behavior_labels) - 1:
    ax.set_xlabel('Filter Size (seconds)')
    #else:
        #ax.set_xlabel('')

    # Legend with friendly names
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [area_labels.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title='Area')

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Save figure
fig.savefig(os.path.join(figures_directory, 'Si_rats_a_mice_time_axis_50nei.png'))


#############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set plotting style
sns.set(style="whitegrid")

# Convert filter size to time in seconds
sampling_rate = 40  # Hz
df_long = df.melt(id_vars=['rat', 'session', 'probe', 'filter', 'behavioral_label'],
                  value_vars=['si', 'si_deep', 'si_sup'],
                  var_name='Area', value_name='SI')
df_long['filter_time'] = df_long['filter'] / sampling_rate

# Define allowed time values (in seconds)
allowed_times = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
df_long = df_long[df_long['filter_time'] <= 1.5]

# Filter DataFrame to only include allowed times and desired behaviors
#df_long = df_long[df_long['filter_time'].isin(allowed_times)]
df_long = df_long[df_long['behavioral_label'].isin(['pos', 'time'])]

# Map custom colors and area names
area_palette = {
    'si': 'black',
    'si_deep': 'gold',
    'si_sup': 'purple'
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
    sns.lineplot(data=sub_df, x='filter_time', y='SI', hue='Area',
                 ax=ax, errorbar='sd', palette=area_palette, marker='o')

    # Formatting
    ax.set_title(f'Structure Index vs Filter Time — {beh}')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_xlabel('Filter Size (seconds)')

    # Legend with friendly names
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [area_labels.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title='Area')

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Save figure
fig.savefig(os.path.join(figures_directory, 'Si_rats_a_mice_time_axis_pos_time_50nei.png'))

###### BAR PLOTA
from scipy.stats import mannwhitneyu

# Filter for 0.2s (filter = 8 at 40 Hz)
df_filtered = df[df['filter'] == 8]
# Melt to long format for plotting
df_long = df_filtered.melt(
    id_vars=['rat', 'session', 'probe', 'behavioral_label'],
    value_vars=['si', 'si_deep', 'si_sup'],
    var_name='Area', value_name='SI_value'
)
# Keep only 'position' and 'time' behavioral labels
df_long = df_long[df_long['behavioral_label'].isin(['time'])]
# Define palette and display names
palette = {'si': 'black', 'si_deep': 'gold', 'si_sup': 'purple'}
area_order = ['si', 'si_deep', 'si_sup']
friendly_labels = {'si': 'Overall', 'si_deep': 'Deep', 'si_sup': 'Superficial'}
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, beh in enumerate(['time']):
    ax = axes[i]
    subset = df_long[df_long['behavioral_label'] == beh]
    sns.barplot(data=subset, x='Area', y='SI_value', palette=palette, order=area_order, ci='sd', ax=ax)
    sns.stripplot(data=subset, x='Area', y='SI_value', order=area_order, color='black', size=6, alpha=0.6, ax=ax)
    # Statistical annotations
    y_max = subset['SI_value'].max()
    height_step = 0.02 * y_max
    base_height = y_max + 0.02
    comparisons = [('si', 'si_deep'), ('si', 'si_sup'), ('si_deep', 'si_sup')]
    for j, (a, b) in enumerate(comparisons):
        data1 = subset[subset['Area'] == a]['SI_value']
        data2 = subset[subset['Area'] == b]['SI_value']
        stat, p = mannwhitneyu(data1, data2)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = None
        if sig:
            x1 = area_order.index(a)
            x2 = area_order.index(b)
            y = base_height + j * height_step
            ax.plot([x1, x1, x2, x2], [y, y + 0.005, y + 0.005, y], lw=1.5, c='k')
            ax.text((x1 + x2) / 2, y + 0.007, sig, ha='center', va='bottom', color='k', fontsize=12)
    ax.set_xticklabels([friendly_labels[a] for a in area_order])
    ax.set_title(f'{beh.capitalize()} SI at 0.2s')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Structure Index (SI)' if i == 0 else '')
    ax.set_xlabel('')
    ax.grid(False)
plt.tight_layout()
plt.show()
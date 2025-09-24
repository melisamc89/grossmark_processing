

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
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

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


def compute_W(patterns):
    """Hopfield-style weight matrix from patterns (time × neurons)."""
    W = np.sum([np.outer(p, p) for p in patterns], axis=0)
    np.fill_diagonal(W, 0)
    return W


def compute_energy(X, W):
    """Hopfield energy over time for signal X (time × neurons)."""
    return -0.5 * np.einsum('ij,ij->i', X @ W, X)


def split_and_energy(data_pre, data, data_post):
    """Handles population-wise energy extraction."""
    energy_dict = {'PRE': {}, 'DATA': {}, 'POST': {}}

    for label, data_full in zip(['pyr', 'sup', 'deep'], data_pre.keys()):
        # Step 1: Use first half of PRE to build W
        X_pre = data_pre[data_full]
        n = X_pre.shape[0]
        W = compute_W(X_pre[:n // 10])

        # Step 2: Compute energy for second half of PRE
        energy_dict['PRE'][label] = compute_energy(X_pre[n // 10:], W)

        # Step 3: Compute energy for DATA and POST using same W
        X_data = data[data_full]
        X_post = data_post[data_full]

        energy_dict['DATA'][label] = compute_energy(X_data, W)
        energy_dict['POST'][label] = compute_energy(X_post, W)

    return energy_dict

def plot_energies(energy_dict):
    for key in ['PRE', 'DATA', 'POST']:
        plt.figure(figsize=(10, 4))
        for label in ['pyr', 'sup', 'deep']:
            plt.plot(energy_dict[key][label], label=label)
        plt.title(f'Energy Over Time ({key})')
        plt.xlabel('Time')
        plt.ylabel('Hopfield Energy')
        plt.legend()
        plt.tight_layout()
        plt.show()
area_palette = {
    'pyr': '#bbbcc0ff',
    'deep': '#cc9900',
    'sup': '#9900ff',
}


def analyze_and_plot_energy_trends(energy_dict, k, save_dir='energy'):
    os.makedirs(save_dir, exist_ok=True)
    summary = []

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    condition_order = ['PRE', 'DATA', 'POST']

    for i, key in enumerate(condition_order):
        ax = axs[i]
        ax.set_title(f'{key} (k={k})')
        ax.set_ylabel('Hopfield Energy')

        for label in ['pyr', 'sup', 'deep']:
            energy = energy_dict[key][label]
            x = np.arange(len(energy))

            # Find 50 local minima (negate for minima)
            peaks, _ = find_peaks(-energy, distance=len(energy) // 50)
            if len(peaks) > 50:
                # Take the lowest 50 minima
                sorted_idx = np.argsort(energy[peaks])[:50]
                minima_x = peaks[sorted_idx]
            else:
                minima_x = peaks

            minima_y = energy[minima_x]

            # Fit linear regression on minima only
            X = minima_x.reshape(-1, 1)
            y = minima_y.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            y_fit = model.predict(X)
            slope = float(model.coef_[0])
            r_squared = r2_score(y, y_fit)
            mean_energy = float(np.mean(energy))

            # Plot full energy curve
            ax.plot(x, energy, label=f'{label} (slope={slope:.3f}, R²={r_squared:.2f})',
                    color=area_palette[label])

            # Plot minima
            ax.scatter(minima_x, minima_y, color=area_palette[label], s=20, marker='x')

            # Plot regression line (across full x-range for visibility)
            x_fit = np.linspace(0, len(energy) - 1, 200).reshape(-1, 1)
            y_line = model.predict(x_fit)
            ax.plot(x_fit, y_line, linestyle='--', color=area_palette[label])

            # Store stats
            summary.append({
                'k': k,
                'condition': key,
                'region': label,
                'slope': slope,
                'r_squared': r_squared,
                'mean_energy': mean_energy
            })

        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'energy_fit_k{k}.svg'))
    plt.savefig(os.path.join(save_dir, f'energy_fit_k{k}.png'))
    plt.close()

    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary)
    return summary_df

def plot_summary_vs_k(summary_df, save_path='energy/summary_vs_k.svg'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics = ['mean_energy', 'slope', 'r_squared']
    titles = ['Mean Energy', 'Slope of Minima Fit', 'R² of Fit']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # Create a line for each combination of condition and region
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for condition in ['PRE', 'DATA', 'POST']:
            for region in ['pyr', 'sup', 'deep']:
                subset = summary_df[(summary_df['condition'] == condition) & (summary_df['region'] == region)]
                if not subset.empty:
                    label = f'{region}-{condition}'
                    ax.plot(subset['k'], subset[metric], marker='o', label=label)
        ax.set_title(titles[i])
        ax.set_xlabel('k (kernel width)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_energy_minima_by_region_all_k(data_summary_dict, save_path='energy/minima_regression_DATA_allfit.svg'):
    """
    Plot energy evolution and regression for DATA condition across all k values,
    with one subplot per region (pyr, deep, sup), and curves for each kernel width.

    Parameters:
    - data_summary_dict: dict with keys as k, values as energy_dicts
    - save_path: file path to save the figure
    """
    area_palette = {
        'pyr': '#bbbcc0ff',
        'deep': '#cc9900',
        'sup': '#9900ff',
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    region_names = ['pyr', 'deep', 'sup']

    for i, region in enumerate(region_names):
        ax = axs[i]
        for k, energy_dict in data_summary_dict.items():
            energy = energy_dict['DATA'][region]
            if len(energy) < 100:  # sanity check
                continue

            # Find local minima (invert signal)
            peaks, _ = find_peaks(-energy, distance=50)
            #if len(peaks) > 50:
            #    peaks = peaks[:50]
            times = np.arange(len(energy))
            y_vals = energy[peaks]

            # Linear regression
            X = peaks.reshape(-1, 1)
            y = y_vals.reshape(-1, 1)
            X = times.reshape(-1, 1)
            y = energy.reshape(-1, 1)
            model = LinearRegression().fit(X, y)

            y_fit = model.predict(X)
            r2 = r2_score(y, y_fit)
            slope = model.coef_[0][0]

            # Plot full energy trace
            ax.plot(times, energy, label=f'k={k}, slope={slope:.3f}, R²={r2:.2f}', alpha=0.6)

            # Overlay detected minima
            #ax.scatter(peaks, y_vals, color='black', s=10)

            # Overlay regression line
            #ax.plot(peaks, y_fit.flatten(), linestyle='--', linewidth=2, color='black')
            ax.plot(times, y_fit.flatten(), linestyle='--', linewidth=2, color='black')

        ax.set_title(region.upper())
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel('Energy')
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle('Energy Minima Regression (DATA condition)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


for rat_index in [0]:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1']:
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode
            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)
            # Define a list of kernel widths to test
            k_values = [5,10,15,20,25,30,35,40,45,50]  # Or any other values you wish to explore
            all_summaries = []

            for k in k_values:
                #### LOAD DATA (BASE)
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                layerID = stimes['LayerID']
                typeID = stimes['TypeID']
                time = np.arange(0, spikes_matrix.shape[0])
                pyr_index = np.array([i for i, ntype in enumerate(typeID) if ntype == 'PYR'])
                deep_index = np.array(
                    [i for i, (v, ntype) in enumerate(zip(layerID, typeID)) if v == 'DEEP' and ntype == 'PYR'])
                superficial_index = np.array(
                    [i for i, (v, ntype) in enumerate(zip(layerID, typeID)) if v == 'SUPERFICIAL' and ntype == 'PYR'])
                data_pyr = data[:, pyr_index]
                data_deep = data[:, deep_index]
                data_sup = data[:, superficial_index]

                #### LOAD DATA PRE
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'PRE_neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                time = np.arange(0, spikes_matrix.shape[0])
                data_pyr_PRE = data[:, pyr_index]
                data_deep_PRE = data[:, deep_index]
                data_sup_PRE = data[:, superficial_index]

                #### LOAD DATA POST
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'POST_neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                time = np.arange(0, spikes_matrix.shape[0])
                data_pyr_POST = data[:, pyr_index]
                data_deep_POST = data[:, deep_index]
                data_sup_POST = data[:, superficial_index]

                # Pack into dictionaries
                data_pre = {'pyr': data_pyr_PRE, 'sup': data_sup_PRE, 'deep': data_deep_PRE}
                data_now = {'pyr': data_pyr, 'sup': data_sup, 'deep': data_deep}
                data_post = {'pyr': data_pyr_POST, 'sup': data_sup_POST, 'deep': data_deep_POST}

                energy_dict = split_and_energy(data_pre, data_now, data_post)

                summary_df = analyze_and_plot_energy_trends(energy_dict, k)
                all_summaries.append(summary_df)
                #energy_dict = split_and_energy(data_pre, data_now, data_post)
                #energy_dict = split_and_energy(data_pre, data_now, data_post)

                # Plot
                #plot_energies(energy_dict, k)
                # Combine and save all results
            final_summary_df = pd.concat(all_summaries, ignore_index=True)
            final_summary_df.to_csv('energy/energy_summary.csv', index=False)


           plot_summary_vs_k(final_summary_df)


            # Dictionary to store results per k
            data_summary_dict = {}
            for k in k_values:
                # Load CURRENT session data
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                layerID = stimes['LayerID']
                typeID = stimes['TypeID']
                pyr_index = np.array([i for i, t in enumerate(typeID) if t == 'PYR'])
                deep_index = np.array(
                    [i for i, (l, t) in enumerate(zip(layerID, typeID)) if l == 'DEEP' and t == 'PYR'])
                superficial_index = np.array(
                    [i for i, (l, t) in enumerate(zip(layerID, typeID)) if l == 'SUPERFICIAL' and t == 'PYR'])
                data_now = {
                    'pyr': data[:, pyr_index],
                    'deep': data[:, deep_index],
                    'sup': data[:, superficial_index]
                }
                # Load PRE session data
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'PRE_neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                data_pre = {
                    'pyr': data[:, pyr_index],
                    'deep': data[:, deep_index],
                    'sup': data[:, superficial_index]
                }
                # Load POST session data
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'POST_neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                with open(spike_file_dir, 'rb') as file:
                    stimes = pkl.load(file)
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                data_post = {
                    'pyr': data[:, pyr_index],
                    'deep': data[:, deep_index],
                    'sup': data[:, superficial_index]
                }

                # Compute energy across PRE, DATA, POST
                energy_dict = split_and_energy(data_pre, data_now, data_post)
                # Store energy_dict in main dictionary
                data_summary_dict[k] = energy_dict

            plot_energy_minima_by_region_all_k(data_summary_dict)







            # --- Bar plot of mean energy with error bars ---
            plt.figure(figsize=(8, 5))
            width = 0.25
            x = np.arange(3)
            for i, cell_type in enumerate(['pyr', 'sup', 'deep']):
                subset = df_energy[df_energy['CellType'] == cell_type]
                plt.bar(x + i * width, subset['MeanEnergy'], yerr=subset['StdEnergy'],
                        width=width, label=cell_type, capsize=4, color=area_palette[cell_type])
            plt.xticks(x + width, ['PRE', 'DATA', 'POST'])
            plt.ylabel('Mean Hopfield Energy')
            plt.title(f'Mean Hopfield Energy by Condition and Cell Type (k={k})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'energy/mean_energy_barplot_k{k}.svg')
            plt.close()

            # --- Regression plot of energy minima ---
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            conditions = ['PRE', 'DATA', 'POST']
            cell_types = ['pyr', 'sup', 'deep']

            for i, condition in enumerate(conditions):
                ax = axs[i]
                ax.set_title(f'Energy Regression - {condition} (k={k})')
                for cell_type in cell_types:
                    energy = energy_dict[condition][cell_type]
                    ax.plot(energy, label=f'{cell_type}', alpha=0.6, color=area_palette[cell_type])
                    peaks, _ = find_peaks(-energy)
                    if len(peaks) > 1:
                        x_peaks = peaks.reshape(-1, 1)
                        y_peaks = energy[peaks]
                        model = LinearRegression().fit(x_peaks, y_peaks)
                        y_pred = model.predict(x_peaks)
                        ax.plot(peaks, y_pred, '--', label=f'{cell_type} fit', color=area_palette[cell_type])
                ax.set_ylabel('Energy')
                ax.legend()
            axs[-1].set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(f'energy/energy_regression_peaks_k{k}.svg')
            plt.close()

            # --- Bar plot of slope ---
            plt.figure(figsize=(8, 5))
            for i, cell_type in enumerate(cell_types):
                subset = df_regression[df_regression['CellType'] == cell_type]
                plt.bar(np.arange(3) + i * width, subset['Slope'], width=width,
                        label=cell_type, color=area_palette[cell_type])
            plt.xticks(np.arange(3) + width, conditions)
            plt.ylabel('Slope')
            plt.title(f'Slope of Energy Minima Fit (k={k})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'energy/slope_energy_minima_k{k}.svg')
            plt.close()

            # --- Bar plot of R² ---
            plt.figure(figsize=(8, 5))
            for i, cell_type in enumerate(cell_types):
                subset = df_regression[df_regression['CellType'] == cell_type]
                plt.bar(np.arange(3) + i * width, subset['R2'], width=width,
                        label=cell_type, color=area_palette[cell_type])
            plt.xticks(np.arange(3) + width, conditions)
            plt.ylabel('R²')
            plt.title(f'R² of Energy Minima Fit (k={k})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'energy/r2_energy_minima_k{k}.svg')
            plt.close()





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





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


for rat_index in [0]:
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1']:
            # --- Define kernel values ---
            k_values = [5, 10, 15, 20, 25, 30]
            # --- Store results ---
            slope_records = []
            r2_records = []

            # --- Area colors ---
            area_palette = {
                'pyr': '#bbbcc0ff',
                'deep': '#cc9900',
                'sup': '#9900ff',
            }
            for k in k_values:
                file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                    session_index] + probe + 'neural_data.pkl'
                spike_file_dir = os.path.join(neural_data_dir, file_name)
                # Open the file in binary read mode
                with open(spike_file_dir, 'rb') as file:
                    # Load the data from the file
                    stimes = pkl.load(file)
                # --- Load base data ---
                spikes_matrix = stimes['spikes_matrix']
                data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
                layerID = stimes['LayerID']
                typeID = stimes['TypeID']
                time = np.arange(0, spikes_matrix.shape[0])
                pyr_index = np.array([i for i, ntype in enumerate(typeID) if ntype == 'PYR'])
                deep_index = np.array(
                    [i for i, (layer, ntype) in enumerate(zip(layerID, typeID)) if layer == 'DEEP' and ntype == 'PYR'])
                superficial_index = np.array([i for i, (layer, ntype) in enumerate(zip(layerID, typeID)) if
                                              layer == 'SUPERFICIAL' and ntype == 'PYR'])
                data_pyr = data[:, pyr_index]
                data_deep = data[:, deep_index]
                data_sup = data[:, superficial_index]

                # Load PRE
                with open(os.path.join(neural_data_dir, file_name + 'PRE_neural_data.pkl'), 'rb') as file:
                    stimes = pkl.load(file)
                data = spikes_to_rates(stimes['spikes_matrix'].T, kernel_width=k)
                data_pyr_PRE = data[:, pyr_index]
                data_deep_PRE = data[:, deep_index]
                data_sup_PRE = data[:, superficial_index]

                # Load POST
                with open(os.path.join(neural_data_dir, file_name + 'POST_neural_data.pkl'), 'rb') as file:
                    stimes = pkl.load(file)
                data = spikes_to_rates(stimes['spikes_matrix'].T, kernel_width=k)
                data_pyr_POST = data[:, pyr_index]
                data_deep_POST = data[:, deep_index]
                data_sup_POST = data[:, superficial_index]

                # Construct data dictionaries
                data_pre = {'pyr': data_pyr_PRE, 'sup': data_sup_PRE, 'deep': data_deep_PRE}
                data_now = {'pyr': data_pyr, 'sup': data_sup, 'deep': data_deep}
                data_post = {'pyr': data_pyr_POST, 'sup': data_sup_POST, 'deep': data_deep_POST}

                # Compute energy
                energy_dict = split_and_energy(data_pre, data_now, data_post)

                # Regression analysis
                conditions = ['PRE', 'DATA', 'POST']
                cell_types = ['pyr', 'sup', 'deep']
                for condition in conditions:
                    for cell_type in cell_types:
                        energy = energy_dict[condition][cell_type]
                        peaks, _ = find_peaks(-energy)
                        if len(peaks) > 1:
                            x_peaks = peaks.reshape(-1, 1)
                            y_peaks = energy[peaks]
                            model = LinearRegression().fit(x_peaks, y_peaks)
                            y_pred = model.predict(x_peaks)
                            r2 = r2_score(y_peaks, y_pred)
                            slope = model.coef_[0]
                            slope_records.append(
                                {'k': k, 'Condition': condition, 'CellType': cell_type, 'Slope': slope})
                            r2_records.append({'k': k, 'Condition': condition, 'CellType': cell_type, 'R2': r2})

                # Save regression plot
                fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                for i, condition in enumerate(conditions):
                    ax = axs[i]
                    ax.set_title(f'Energy Regression - {condition} (k={k})')
                    for cell_type in cell_types:
                        energy = energy_dict[condition][cell_type]
                        ax.plot(energy, label=cell_type, alpha=0.6, color=area_palette[cell_type])
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
                plt.savefig(f'energy/energy_regression_peaks_k{k}.png')
                plt.savefig(f'energy/energy_regression_peaks_k{k}.svg')
                plt.close()

            # --- Save results as CSV ---
            df_slope = pd.DataFrame(slope_records)
            df_r2 = pd.DataFrame(r2_records)
            df_slope.to_csv('energy/slope_vs_k.csv', index=False)
            df_r2.to_csv('energy/r2_vs_k.csv', index=False)

            # --- Plot slope vs k ---
            plt.figure(figsize=(8, 5))
            for cell_type in cell_types:
                subset = df_slope[df_slope['CellType'] == cell_type]
                means = subset.groupby('k')['Slope'].mean()
                plt.plot(means.index, means.values, label=cell_type, marker='o')
            plt.xlabel('Kernel Width (k)')
            plt.ylabel('Mean Slope')
            plt.title('Slope of Energy Minima vs. Kernel Width')
            plt.legend()
            plt.tight_layout()
            plt.savefig('energy/slope_vs_k.png')
            plt.savefig('energy/slope_vs_k.svg')
            plt.close()

            #############################3
            k = 25
            spikes_matrix = stimes['spikes_matrix']
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
            layerID = stimes['LayerID']
            typeID = stimes['TypeID']
            time = np.arange(0,spikes_matrix.shape[0])
            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])
            # Find indices where the value is 'DEEP'
            deep_index = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'PYR'])
            # Find indices where the value is 'SUPERFICIAL'
            superficial_index = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'PYR'])
            data_deep= data[:,deep_index]
            data_sup =  data[:,superficial_index]
            data_pyr=data[:,pyr_index]

            ###### LOAD DATA PRE
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] + probe + 'PRE_neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode
            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)
            spikes_matrix = stimes['spikes_matrix']
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
            time = np.arange(0, spikes_matrix.shape[0])
            data_deep_PRE = data[:, deep_index]
            data_sup_PRE = data[:, superficial_index]
            data_pyr_PRE = data[:, pyr_index]


            ###### LOAD DATA POST
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] + probe + 'POST_neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode
            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)
            spikes_matrix = stimes['spikes_matrix']
            data = spikes_to_rates(spikes_matrix.T, kernel_width=k)
            time = np.arange(0, spikes_matrix.shape[0])
            data_deep_POST = data[:, deep_index]
            data_sup_POST = data[:, superficial_index]
            data_pyr_POST = data[:, pyr_index]

            # Assume the data variables are already defined from your loading code:
            data_pre = {
                'pyr': data_pyr_PRE,
                'sup': data_sup_PRE,
                'deep': data_deep_PRE
            }

            data_now = {
                'pyr': data_pyr,
                'sup': data_sup,
                'deep': data_deep
            }

            data_post = {
                'pyr': data_pyr_POST,
                'sup': data_sup_POST,
                'deep': data_deep_POST
            }

            energy_dict = split_and_energy(data_pre, data_now, data_post)

            import os
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import r2_score
            from scipy.signal import find_peaks
            from sklearn.linear_model import LinearRegression

            # Ensure subdirectory exists
            os.makedirs('energy', exist_ok=True)

            area_palette = {
                'pyr': '#bbbcc0ff',
                'deep': '#cc9900',
                'sup': '#9900ff',
            }


            # --- Line plot of energy over time ---
            # --- Line plot of energy over time ---
            def plot_energies(energy_dict, k):
                for key in ['PRE', 'DATA', 'POST']:
                    plt.figure(figsize=(10, 4))
                    for label in ['pyr', 'sup', 'deep']:
                        plt.plot(energy_dict[key][label], label=label, color=area_palette[label])
                    plt.title(f'Energy Over Time ({key}, k={k})')
                    plt.xlabel('Time')
                    plt.ylabel('Hopfield Energy')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'energy/energy_over_time_{key.lower()}_k{k}.svg')
                    plt.close()


            plot_energies(energy_dict, k)

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



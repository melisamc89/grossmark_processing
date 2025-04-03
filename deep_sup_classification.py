from turtledemo.nim import computerzug

import numpy as np
import os
import pickle as pkl

import matplotlib.pyplot as plt
import copy
from src.general_utils import *
from src.lfp_analysis import rat_directory
from src.plotting_utils import *
import scipy as sp
from src.channel_information import *

### declare general paths and folders structure to the data
data_directory = '/home/melma31/Downloads/Grossmark'
base_directory = '/home/melma31/Documents/time_project'
files_directory = os.path.join(base_directory, 'output')
files_directory = os.path.join(base_directory, 'classification')
figures_directory = os.path.join(base_directory, 'figures')

rat_names = ['Achilles','Buddy','Cicero','Gatsby']
rat_sessions = {}
rat_sessions['Achilles'] = ['10252013','11012013']
rat_sessions['Buddy'] = ['06272013']
rat_sessions['Cicero'] = ['09012014','09102014','09172014']
rat_sessions['Gatsby'] = ['08022013','08282013']

ripple_file_extension = '_ripple_power_output.pkl'
waveform_file_extension = '_waveform_output.pkl'
theta_file_extension = '_theta_output.pkl'

rat_index = 0
rat_session = 0

ripple_file_dir = os.path.join(base_directory, 'output', rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session]
                               + ripple_file_extension)
waveform_file_dir= os.path.join(base_directory, 'output', rat_names[rat_index] + '_' +rat_sessions[rat_names[rat_index]][rat_session]
                               + waveform_file_extension)
# Open the file in binary read mode
with open(ripple_file_dir, 'rb') as file:
    # Load the data from the file
    ripple = pkl.load(file)
with open(waveform_file_dir, 'rb') as file:
    # Load the data from the file
    waveform = pkl.load(file)

neuron_classification = {}
for probe in waveform.keys():
    neuron_classification[probe] = {}
    for shank in waveform[probe].keys():
        neuron_classification[probe][shank] = {}
        neuron_class_list = []
        cluster_id_list = []
        for neuron in range(waveform[probe][shank]['waveform'].shape[0]):
            neuron_waveform = waveform[probe][shank]['waveform'][neuron,:,:]
            cluster_id = waveform[probe][shank]['cluster_id'][neuron]
            peak_to_peak = -np.min(neuron_waveform, axis=0) + np.max(neuron_waveform, axis=0)
            max_index = np.argmax(peak_to_peak)
            power_spectrum = ripple[probe][shank]['power_spectrum']
            max_power_spectrum_channel =ripple[probe][shank]['channel_information']['max_ripple_power_channel']
            if max_index >= max_power_spectrum_channel:
                classification = 'deep'
            else:
                classification = 'superficial'
            neuron_class_list.append(classification)
            cluster_id_list.append(cluster_id)
            #fig = plot_waveform_and_power(neuron_waveform, power_spectrum, max_power_spectrum_channel, max_index,
            #                              classification)

            #savedir = figures_directory+'/'+rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session]
            #if not os.path.exists(savedir):
            #    os.makedirs(savedir)

            #file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session] +'_' + probe + '_' + shank + '_' + str(waveform[probe][shank]['cluster_id'][neuron]) + '.jpg'
            #file_directory = os.path.join(savedir, file_name)
            #fig.savefig(file_directory)
            #plt.close()
        neuron_classification[probe][shank]['classification'] = neuron_class_list
        neuron_classification[probe][shank]['neuron_id'] = cluster_id_list

# Define the filename where the dictionary will be stored
output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][rat_session] + '_neuron_classification_output.pkl'
# Open the file for writing in binary mode and dump the dictionary
with open(os.path.join(files_directory , output_filename), 'wb') as file:
    pkl.dump(neuron_classification, file)

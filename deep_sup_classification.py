import pickle as pkl
from src.plotting_utils import *
from src.channel_information import*
from src.config import *


ripple_file_extension = '_ripple_power_output.pkl'
waveform_file_extension = '_waveform_output.pkl'
theta_file_extension = '_theta_output.pkl'

sessions = [[0,1],[0],[0,1,2],[0,1]]

for rat_index in range(0,4):
    print('Classifing neurons from rat: ', rat_names[rat_index])
    for rat_session in sessions[rat_index]:
        print('Session Number ... ', rat_session + 1)
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
                    if max_index > max_power_spectrum_channel:
                        classification = 'SUPERFICIAL'
                    else:
                        classification = 'DEEP'
                    neuron_class_list.append(classification)
                    cluster_id_list.append(cluster_id)
                    fig = plot_waveform_and_power(neuron_waveform, power_spectrum, max_power_spectrum_channel, max_index,
                                                  classification)

                    savedir = figures_directory+'/'+rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session]
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session] +'_' + probe + '_' + shank + '_' + str(waveform[probe][shank]['cluster_id'][neuron]) + '.jpg'
                    file_directory = os.path.join(savedir, file_name)
                    fig.savefig(file_directory)
                    plt.close(fig)
                neuron_classification[probe][shank]['classification'] = neuron_class_list
                neuron_classification[probe][shank]['neuron_id'] = cluster_id_list

        neuron_classification['ripple_range'] = [150,200]
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][rat_session] + '_neuron_classification_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(files_directory , output_filename), 'wb') as file:
            pkl.dump(neuron_classification, file)

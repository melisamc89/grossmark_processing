import pickle as pkl
from src.plotting_utils import *
from src.channel_information import*
from src.config import *
from src.general_utils import *


ripple_file_extension = '_ripple_power_output.pkl'
waveform_file_extension = '_waveform_output.pkl'
theta_file_extension = '_theta_output.pkl'

from scipy.signal import butter, filtfilt, hilbert, welch

# === Filtering functions ===
def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def classify_deep_sup_2(neuron_waveform, power_spectrum, low_freq= 150, high_freq = 250):
    peak_to_peak = -np.min(neuron_waveform, axis=0) + np.max(neuron_waveform, axis=0)
    max_index = np.argmax(peak_to_peak)
    max_power_spectrum_channel = ripple[probe][shank]['channel_information'][
        'max_ripple_power_channel']
    area_vect = []
    for channel in range(power_spectrum['power'].shape[1]):
        frequencies = power_spectrum['frequency']
        power_values = power_spectrum['power'][:, channel]
        normalized_power = power_values / np.max(power_values)
        # Highlight the area under the curve between 100 Hz and 250 Hz
        idx = (frequencies >= low_freq) & (frequencies <= high_freq)
        # Calculate and display the area under the curve
        area = np.sum(normalized_power[idx])
        area_vect.append(area)
    sorted_area = np.sort(area_vect)[::-1]
    sorted_area_index = np.argsort(area_vect)[::-1]
    diff_sorted_area = np.abs(np.diff(sorted_area))
    #if diff_sorted_area[0] < 0.005:
    #    max_power_spectrum_channel = min(sorted_area_index[0], sorted_area_index[1])
    max_power_spectrum_channel = sorted_area_index[0]
    classification = 'BORDER'
    if max_index > max_power_spectrum_channel:
        classification = 'SUPERFICIAL'
    else:
        classification = 'DEEP'
    return classification, max_power_spectrum_channel, max_index


def classify_deep_sup(neuron_waveform, raw, sf = 1000, low_freq= 150, high_freq = 250):

    ripple_filtered = bandpass_filter(raw, sf, low_freq, high_freq)
    ripple_power = np.sum(ripple_filtered ** 2, axis=0)
    max_power_spectrum_channel = np.argmax(ripple_power)

    peak_to_peak = -np.min(neuron_waveform, axis=0) + np.max(neuron_waveform, axis=0)
    max_index = np.argmax(peak_to_peak)
    #max_power_spectrum_channel = ripple[probe][shank]['channel_information'][
    #    'max_ripple_power_channel']

    classification = 'BORDER'
    if max_index > max_power_spectrum_channel:
        classification = 'SUPERFICIAL'
    else:
        classification = 'DEEP'
    return classification, max_power_spectrum_channel, max_index


for rat_index in [0,1,2,3]:
    print('Classifing neurons from rat: ', rat_names[rat_index])
    low_freq = low_ripple_freq[rat_index]
    high_freq = high_ripple_freq[rat_index]
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

        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][rat_session] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)
        session_info = convert_session_mat_to_dict(session_information_directory)

        spikes_ids = session_info['spikes']['spikes_ids']
        n_neurons = len(np.unique(spikes_ids))
        spikes_ids_shank = np.floor(spikes_ids / 100)
        spikes_id_cluster = np.mod(spikes_ids, 100)

        pyr_ids = session_info['spikes']['pyr_ids']
        int_ids = session_info['spikes']['int_ids']
        pyr_ids_shank = np.floor(pyr_ids / 100)
        pyr_id_cluster = np.mod(pyr_ids, 100)

        int_ids_shank = np.floor(int_ids / 100)
        int_id_cluster = np.mod(int_ids, 100)

        neuron_classification = {}
        for probe in waveform.keys():
            neuron_classification[probe] = {}
            for shank in waveform[probe].keys():
                neuron_classification[probe][shank] = {}
                neuron_class_list = []
                cluster_id_list = []

                spkGroup = waveform[probe][shank]['Spk.Group']

                spkGroup_index = np.where(spikes_ids_shank[0,:] == spkGroup)[0]
                spkGroup_cells = spikes_id_cluster[0,spkGroup_index]
                n_neurons_shank = len(np.unique(spkGroup_cells))

                spkGroupPyr_index = np.where(pyr_ids_shank[0,:] == spkGroup)[0]
                spkGroupPyr_cells = pyr_id_cluster[0,spkGroupPyr_index]

                spkGroupInt_index = np.where(int_ids_shank[0,:] == spkGroup)[0]
                spkGroupInt_cells = int_id_cluster[0,spkGroupInt_index]

                pyr_int_class_list = []
                for neuron in range(waveform[probe][shank]['waveform'].shape[0]):
                    neuron_waveform = waveform[probe][shank]['waveform'][neuron,:,:]
                    cluster_id = waveform[probe][shank]['cluster_id'][neuron]
                    power_spectrum = ripple[probe][shank]['power_spectrum']
                    lfp = ripple[probe][shank]['lfp']
                    classification, max_power_spectrum_channel, max_index = classify_deep_sup(neuron_waveform, lfp ,1000, low_freq,high_freq)
                    neuron_class_list.append(classification)
                    cluster_id_list.append(cluster_id)
                    if neuron in spkGroupPyr_cells:
                        pyr_int_class_list.append('PYR')
                        neuron_type = 'PYRAMIDAL'
                    else:
                        if neuron in spkGroupInt_cells:
                            pyr_int_class_list.append('INT')
                            neuron_type = 'INTERNEURON'
                        else:
                            pyr_int_class_list.append('NoCat')
                            neuron_type = 'No Category'

                    fig = plot_waveform_and_power(neuron_waveform, power_spectrum, max_power_spectrum_channel, max_index,
                                                  classification,low_freq,high_freq,neuron_type)

                    savedir = figures_directory+'/'+rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session]
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][rat_session] +'_' + probe + '_' + shank + '_' + str(waveform[probe][shank]['cluster_id'][neuron]) + '.svg'
                    file_directory = os.path.join(savedir, file_name)
                    fig.savefig(file_directory)
                    plt.close(fig)
                neuron_classification[probe][shank]['classification'] = neuron_class_list
                neuron_classification[probe][shank]['neuron_id'] = cluster_id_list
                neuron_classification[probe][shank]['neuron_type'] = pyr_int_class_list

        neuron_classification['ripple_range'] = [150,250]
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][rat_session] + '_neuron_classification_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        if not os.path.exists(files_directory):
            os.makedirs(files_directory)
        # Define the filename where the dic
        with open(os.path.join(files_directory , output_filename), 'wb') as file:
            pkl.dump(neuron_classification, file)

from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *
import pickle as pkl

sessions = [[0,1],[0],[0,1,2],[0,1]]

for rat_index in range(0,4):
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        #novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        #xml_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.xml'
        #xml_file_directory = os.path.join(rat_directory, session_directory,xml_file_name)
        #session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        #session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        #session_info = convert_session_mat_to_dict(session_information_directory)
        #params = get_recorging_parameters(xml_file_directory)
        # Accessing data for Rat A, Session 1, Probe 1, Shank 2
        channels = channel_organization[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][session_index]]
        class_file_extension = '_neuron_classification_output.pkl'
        classification_file_dir = os.path.join(base_directory, 'classification', rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index]
                                       + class_file_extension)
        with open(classification_file_dir, 'rb') as file:
            # Load the data from the file
            classification = pkl.load(file)
        spikes_times_dict = {}
        for probe in channels.keys():
            number_of_shanks = len(channels[probe]['Spk.Group'])
            spikes_times_dict[probe] = {}
            neurons_spikes_list = []
            classification_list = []
            for shank in channels[probe]['Spk.Group'].keys():
                if shank not in discarded_shanks[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][session_index]][probe]:
                    spk_group = channels[probe]['Spk.Group'][shank]
                    spk_times_filename = os.path.join(rat_directory,session_directory,rat_names[rat_index]
                                            +'_'+rat_sessions[rat_names[rat_index]][session_index]+'.res.' + str(spk_group))
                    spikes_times = read_spikes_times(spk_times_filename)
                    filename_clu = os.path.join(rat_directory,session_directory,rat_names[rat_index]+
                                                '_'+rat_sessions[rat_names[rat_index]][session_index]+'.clu.'+ str(spk_group))
                    num_clusters, cluster_ids = read_klusters_clu_file(filename_clu)
                    clusters_names = np.unique(cluster_ids)
                    for cluster in clusters_names:
                        if cluster!=0 and cluster!=1:
                            neuron_location = np.where(cluster_ids == cluster)
                            neuron_spikes = spikes_times[neuron_location]
                            neurons_spikes_list.append(neuron_spikes)
                            neuron_id_class_index = np.where(classification[probe][shank]['neuron_id']==cluster)[0][0]
                            neuron_classification = classification[probe][shank]['classification'][neuron_id_class_index]
                            classification_list.append(neuron_classification)
            spikes_times_dict[probe]['stimes'] = neurons_spikes_list
            spikes_times_dict[probe]['classification'] = classification_list

        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_stimes_classified.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(files_directory , output_filename), 'wb') as file:
            pkl.dump(spikes_times_dict, file)
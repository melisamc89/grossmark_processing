from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import*

sessions = [[0,1],[0],[0,1,2],[0,1]]
for rat_index in [3]:
    print('Extraction Spikes Clusters: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)

        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        #xml_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.xml'
        #xml_file_directory = os.path.join(rat_directory, session_directory,xml_file_name)
        #session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        #session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        #session_info = convert_session_mat_to_dict(session_information_directory)
        #params = get_recorging_parameters(xml_file_directory)
        # Accessing data for Rat A, Session 1, Probe 1, Shank 2
        channels = channel_organization[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][session_index]]

        unmatched_counter = 0
        neurons_counter = {}
        waveform_dict = {}
        for probe in channels.keys():
            number_of_shanks = len(channels[probe]['Spk.Group'])
            probe_neurons_counts = 0
            waveform_dict[probe] = {}
            for shank in channels[probe]['Spk.Group'].keys():
                waveform_dict[probe][shank] = {}
                spk_group = channels[probe]['Spk.Group'][shank]
                spk_filename = os.path.join(rat_directory,session_directory,rat_names[rat_index]
                                        +'_'+rat_sessions[rat_names[rat_index]][session_index]+'.spk.' + str(spk_group))
                number_of_channels = len(channels[probe][shank])
                waveform = read_spike_waveforms(spk_filename, number_of_channels, samples_per_spike=32)
                filename_clu = os.path.join(rat_directory,session_directory,rat_names[rat_index]+
                                            '_'+rat_sessions[rat_names[rat_index]][session_index]+'.clu.'+ str(spk_group))
                num_clusters, cluster_ids = read_klusters_clu_file(filename_clu)

                clusters_names = np.unique(cluster_ids)
                #print("Number of clusters:", len(clusters_names))
                #print("Clusters_ID:", clusters_names)
                number_of_spikes = waveform.shape[0]
                number_of_clusters_id = len(cluster_ids)
                #print('Number of spikes: ', number_of_spikes)
                #print('Number of clusters: ', number_of_clusters_id)
                #################### VERIFY THIS ###############################3
                if number_of_spikes != number_of_clusters_id:
                    print('Number of clusters does not match number of spikes')
                mean_waveform = np.zeros((len(clusters_names),32,number_of_channels))
                for index, i in enumerate(clusters_names):
                    x = np.where(cluster_ids == i)[0]
                    mean_waveform[index,:,:] = np.mean(waveform[x,:,:], axis=0)
                waveform_dict[probe][shank]['waveform'] = mean_waveform
                waveform_dict[probe][shank]['cluster_id'] = clusters_names
                probe_neurons_counts = probe_neurons_counts + len(clusters_names)
            neurons_counter[probe] =  probe_neurons_counts

        import pickle as pkl
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_waveform_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(output_directory , output_filename), 'wb') as file:
            pkl.dump(waveform_dict, file)

        print('Total number of neurons in ' + probe + ' : ' + str (neurons_counter))

#plot_single_spike_waveform(waveform_dict['Probe1']['LeftCA1Shank5']['waveform'], spike_index=0, color_map='viridis')


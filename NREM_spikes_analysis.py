from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *
import pickle as pkl

novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
theta_file_extension = '_theta_output.pkl'

for rat_index in range(0,4):
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)

        spktimes_file_name = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_stimes_classified.pkl'

        spktimes_file_dir= os.path.join(base_directory, 'classification', spktimes_file_name)


        # Open the file in binary read mode
        with open(spktimes_file_dir, 'rb') as file:
            # Load the data from the file
            stimes = pkl.load(file)

        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)
        session_info = convert_session_mat_to_dict(session_information_directory)

        NREM_durections = session_info['epochs']['NREMEpoch'][1,:] - session_info['epochs']['NREMEpoch'][0,:]
        NREM_max_duration = np.max(NREM_durections)
        NREM_max_duration_index = np.argmax(NREM_durections)

        start_resting = int(session_info['epochs']['NREMEpoch'][0,NREM_max_duration_index]  )
        end_resting = int(session_info['epochs']['NREMEpoch'][1,NREM_max_duration_index])

        start_maze = start_resting
        end_maze = end_resting
        start_maze = session_info['epochs']['preEpoch'][0][0]
        end_maze = session_info['epochs']['postEpoch'][1][0]

        positionTimeStamps = np.arange(start_maze, end_maze, 1 / 40)

        for probe in stimes.keys():
            output_dic = {}
            spikes_probe = stimes[probe]['stimes']
            spike_matrix = np.zeros((len(positionTimeStamps),len(spikes_probe)))
            for neuron in range(len(spikes_probe)):
                neuron_spikes = spikes_probe[neuron]
                maze_spikes_index = np.logical_and(neuron_spikes > start_maze * 20000, neuron_spikes < end_maze * 20000 )
                maze_spikes = spikes_probe[neuron][maze_spikes_index]/20000
                # Find the index in timestamps_a for each event in events_b
                indices = np.searchsorted(positionTimeStamps, maze_spikes, side='right') - 1
                spike_matrix[indices,neuron] = 1
            output_dic['spikes_matrix'] = spike_matrix
            output_dic['LayerID'] = stimes[probe]['classification']
            output_dic['TypeID'] = stimes[probe]['type']



            data_output_directory = '/home/melma31/Documents/time_project/data'
            # Define the filename where the dictionary will be stored
            output_filename = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] + probe + 'POST_neural_data.pkl'
            # Open the file for writing in binary mode and dump the dictionary
            with open(os.path.join(files_directory, output_filename), 'wb') as file:
                pkl.dump(output_dic, file)

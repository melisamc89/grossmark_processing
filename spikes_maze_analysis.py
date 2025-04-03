from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *
import pickle as pkl
sessions = [[0,1],[0],[0,1,2],[0,1]]

novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
theta_file_extension = '_theta_output.pkl'

for rat_index in range(0,4):
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)

        spktimes_file_name = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_stimes_classified.pkl'

        spktimes_file_dir= os.path.join(base_directory, 'classification', spktimes_file_name)

        theta_file_dir = os.path.join(base_directory, 'output', rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index]
                                       + theta_file_extension )

        # Open the file in binary read mode
        with open(spktimes_file_dir, 'rb') as file:
            # Load the data from the file
            stimes = pkl.load(file)

        # Open the file in binary read mode
        with open(theta_file_dir, 'rb') as file:
            # Load the data from the file
            theta = pkl.load(file)

        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)
        session_info = convert_session_mat_to_dict(session_information_directory)
        start_maze = session_info['epochs']['mazeEpoch'][0][0]
        end_maze = session_info['epochs']['mazeEpoch'][1][0]

        preposition1D = session_info['position']['position1D']
        not_nan = ~np.isnan(preposition1D)
        position1D = preposition1D[not_nan].reshape(-1)
        positionTimeStamps = np.hstack(session_info['position']['time_staps'])
        positionTimeStamps= positionTimeStamps[not_nan[0]]

        for probe in stimes.keys():
            theta_signal = theta[probe]['theta']
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
            output_dic['position'] = position1D
            output_dic['theta'] = theta_signal

            data_output_directory = '/home/melma31/Documents/time_project/data'
            # Define the filename where the dictionary will be stored
            output_filename = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] + probe + 'neural_data.pkl'
            # Open the file for writing in binary mode and dump the dictionary
            with open(os.path.join(files_directory, output_filename), 'wb') as file:
                pkl.dump(output_dic, file)

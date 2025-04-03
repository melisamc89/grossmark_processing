from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *
import pickle as pkl
sessions = [[0,1],[0],[0,1,2],[0,1]]

novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')

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
        start_maze = session_info['epochs']['mazeEpoch'][0][0]
        end_maze = session_info['epochs']['mazeEpoch'][1][0]

        preposition1D = session_info['position']['position1D']
        not_nan = ~np.isnan(preposition1D)
        position1D = preposition1D[not_nan]
        for probe in stimes.keys():
            spikes_probe = stimes[probe]['stimes']
            for neuron in len(spikes_probe):
                neuron_spikes = spikes_probe[neuron]/20000
                maze_spikes_index = np.logical_and(neuron_spikes > start_maze, neuron_spikes < end_maze)
                maze_spikes = neuron_spikes[maze_spikes_index] - start_maze
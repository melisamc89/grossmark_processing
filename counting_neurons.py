import pandas as pd
import pickle as pkl
from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *
### print session type ####

for rat_index in range(0,4):
    #print('Extraction Ripple Bands from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        #print('Session Number ... ', session_index + 1)
        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        session_info = convert_session_mat_to_dict(session_information_directory)
        maze_type = session_info['position']['maze_type']
        array = maze_type
        # Convert the array to a list of characters
        characters = [chr(code) for code in array.flatten()]
        # Join the list of characters into a string
        maze_string = ''.join(characters)
        print('Rat ' + rat_names[rat_index] + ' in session' + str(session_index + 1) + ' is ' + maze_string)



neural_data_dir = files_directory

rat_name = []
rat_probe = []
rat_session = []
total_neurons = []

total_pyr = []
total_int = []
total_nn = []

total_deep = []
total_sup = []

total_pyr_deep = []
total_int_deep = []
total_nn_deep = []

total_pyr_sup = []
total_int_sup = []
total_nn_sup = []


for rat_index in range(0,4):
    print('Extracting spikes times from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        for probe in ['Probe1','Probe2']:
            file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][session_index] +probe +'neural_data.pkl'
            spike_file_dir = os.path.join(neural_data_dir, file_name)
            # Open the file in binary read mode
            with open(spike_file_dir, 'rb') as file:
                # Load the data from the file
                stimes = pkl.load(file)

            # print('Session Number ... ', session_index + 1)
            rat_directory = os.path.join(data_directory, rat_names[rat_index])
            session_directory = rat_names[rat_index] + '_' + str(rat_sessions[rat_names[rat_index]][session_index])
            novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
            session_information_file_name = rat_names[rat_index] + '_' + rat_sessions[rat_names[rat_index]][
                session_index] + '_sessInfo.mat'
            session_information_directory = os.path.join(data_directory, novelty_session_directory,
                                                         session_information_file_name)
            #### load session info, parameters, and lfp
            session_info = convert_session_mat_to_dict(session_information_directory)
            maze_type = session_info['position']['maze_type']
            array = maze_type
            # Convert the array to a list of characters
            characters = [chr(code) for code in array.flatten()]
            # Join the list of characters into a string
            maze_string = ''.join(characters)
            #print('Rat ' + rat_names[rat_index] + ' in session' + str(session_index + 1) + ' is ' + maze_string)

            spikes_matrix = stimes['spikes_matrix']
            total_neurons.append(spikes_matrix.shape[1])
            layerID = stimes['LayerID']
            typeID = stimes['TypeID']
            rat_name.append(rat_names[rat_index])
            rat_probe.append(probe)
            rat_session.append(maze_string)

            # Find indices where the value is 'DEEP'
            deep_index_pyr = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'PYR'])
            total_pyr_deep.append(len(deep_index_pyr))
            deep_index_pyr = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'INT'])
            total_int_deep.append(len(deep_index_pyr))
            deep_index_pyr = np.array([index for index, (value, ntype) in enumerate(zip(layerID, typeID)) if
                                   value == 'DEEP' and ntype == 'NoCat'])
            total_nn_deep.append(len(deep_index_pyr))

            superficial_index_pyr = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'PYR'])
            total_pyr_sup.append(len(superficial_index_pyr))
            superficial_index_pyr = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'INT'])
            total_int_sup.append(len(superficial_index_pyr))

            superficial_index_pyr = np.array([index for index, (value, ntype)  in enumerate(zip(layerID, typeID)) if
                                          value == 'SUPERFICIAL' and ntype == 'NoCat'])
            total_nn_sup.append(len(superficial_index_pyr))

            pyr_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'PYR'])
            total_pyr.append(len(pyr_index))
            int_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'INT'])
            total_int.append(len(int_index))
            # Find indices where the value is 'SUPERFICIAL'
            nn_index = np.array([index for index, ntype in enumerate(typeID) if
                                   ntype == 'NoCat'])
            total_nn.append(len(nn_index))


            nn_index = np.array([index for index, ntype in enumerate(layerID) if
                                   ntype == 'DEEP'])
            total_deep.append(len(nn_index))

            nn_index = np.array([index for index, ntype in enumerate(layerID) if
                                   ntype == 'SUPERFICIAL'])
            total_sup.append(len(nn_index))



final_dict = {
    'Name': rat_name,
    'Probe': rat_probe,
    'Session': rat_session,
    'total_neurons': total_neurons,

    'total_pyr': total_pyr,
    'total_int': total_int,
    'total_nn': total_nn,

    'total_deep': total_deep,
    'total_sup': total_sup,

    'total_pyr_deep': total_pyr_deep,
    'total_int_deep': total_int_deep,
    'total_nn_deep': total_nn_deep,

    'total_pyr_sup': total_pyr_sup,
    'total_int_sup': total_int_sup,
    'total_nn_sup': total_nn_sup,
}

output_pd = pd.DataFrame(final_dict)
output_pd.to_csv(os.path.join(output_directory,'total_neurons_new_criteria.csv'))


# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(14, 4))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')
# Create the table
table = ax.table(cellText=output_pd.values, colLabels=output_pd.columns, cellLoc = 'center', loc='center')
# Adjust table scaling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # May need adjustment to fit your screen
plt.show()
fig.savefig(figures_directory + 'total_neuron_new_criteria.png')
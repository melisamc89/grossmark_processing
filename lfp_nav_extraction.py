
from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import *

for rat_index in range(0):
    print('Extraction Ripple Bands from rat: ', rat_names[rat_index])
    for session_index in sessions[rat_index]:
        print('Session Number ... ', session_index + 1)
        rat_directory = os.path.join(data_directory, rat_names[rat_index])
        session_directory = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index])
        novelty_session_directory = os.path.join(data_directory, 'NoveltySessInfoMatFiles')
        eef_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.eeg'
        xml_file_name = rat_names[rat_index]+'_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.xml'
        eeg_file_directory = os.path.join(rat_directory, session_directory,eef_file_name)
        xml_file_directory = os.path.join(rat_directory, session_directory,xml_file_name)
        session_information_file_name =  rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_sessInfo.mat'
        session_information_directory = os.path.join(data_directory,novelty_session_directory, session_information_file_name)

        #### load session info, parameters, and lfp
        session_info = convert_session_mat_to_dict(session_information_directory)
        params = get_recorging_parameters(xml_file_directory)
        #### general variables
        lfp_sr = params['lfpSamplingRate']
        target_fs = lfp_sr
        lfp = np.fromfile(eeg_file_directory, dtype='int16')
        lfp = lfp.reshape((-1, number_of_channels[rat_names[rat_index]]))
        navigation_period = session_info['epochs']['mazeEpoch'][1] - session_info['epochs']['mazeEpoch'][0]
        start_resting = int(session_info['epochs']['mazeEpoch'][0] * lfp_sr )
        end_resting = int(session_info['epochs']['mazeEpoch'][1] * lfp_sr )
        navigation_signal = get_signals_segment(lfp, start_resting, end_resting)
        del lfp
        #target_fs = 500  # Target sampling rate in Hz
        #downsampled_lfp= downsample_signal(navigation_signal, lfp_sr, target_fs)
        #del navigation_signal
        # Accessing data for Rat A, Session 1, Probe 1, Shank 2
        channels = channel_organization[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][session_index]]
        #### process channels and signals and extract resting period
        shank_signals = group_lfp_by_shank(navigation_signal, channels)
        del navigation_signal
        spk_group = {}
        spk_group_probe_1 = channels['Probe1']['Spk.Group']
        spk_group_probe_2 = channels['Probe2']['Spk.Group']
        spk_group['Probe1'] = spk_group_probe_1
        spk_group['Probe2'] = spk_group_probe_2

        combined_dict = {}
        output_dict = {}
        for probe in spk_group:
            output_dict[probe] = {}
            for shank in spk_group[probe]:
                #ripple_power_channel = combined_dict[probe][shank]['max_ripple_power_channel']
                #theta_power_channel = combined_dict[probe][shank]['max_theta_power_channel']
                output_dict[probe][shank] = {
                    'channel_information': spk_group[probe][shank],
                    'lfp': shank_signals[probe][shank],
                }
        import pickle as pkl
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_lfp_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(output_directory , output_filename), 'wb') as file:
            pkl.dump(output_dict, file)
        del output_dict
        del combined_dict
        del shank_signals

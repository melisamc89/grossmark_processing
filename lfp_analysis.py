from src.general_utils import *
from src.plotting_utils import *
from src.channel_information import *
from src.config import*

### will later create a loop on this
rat_index = 0
sessions = [[0,1],[0],[0,1,2],[0,1]]
for rat_index in range(0,4):
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


        #### start loading
        lfp_sr = params['lfpSamplingRate']
        lfp = np.fromfile(eeg_file_directory, dtype='int16')
        lfp = lfp.reshape((-1, number_of_channels[rat_names[rat_index]]))

        ### get NREM period
        NREM_durections = session_info['epochs']['NREMEpoch'][1,:] - session_info['epochs']['NREMEpoch'][0,:]
        NREM_max_duration = np.max(NREM_durections)
        NREM_max_duration_index = np.argmax(NREM_durections)

        start_resting = int(session_info['epochs']['NREMEpoch'][0,NREM_max_duration_index] * lfp_sr )
        end_resting = int(session_info['epochs']['NREMEpoch'][1,NREM_max_duration_index] * lfp_sr )
        NREM_signal = get_signals_segment(lfp, start_resting, end_resting)

        del lfp
        target_fs = 1250  # Target sampling rate in Hz
        downsampled_lfp= downsample_signal(NREM_signal, lfp_sr, target_fs)
        del NREM_signal

        # Accessing data for Rat A, Session 1, Probe 1, Shank 2
        channels = channel_organization[rat_names[rat_index]][rat_sessions[rat_names[rat_index]][0]]
        #### process channels and signals and extract resting period
        shank_signals = group_lfp_by_shank(downsampled_lfp, channels)
        del downsampled_lfp
        ################################################
        ####        PLOTTING SINGALS #########
        ################################################
        ### only plot for short signals
        #plot_probe_signals(shank_signals)
        #################################################
        #### COMPUTE POWER SPECTRA (and plot)               ########
        #################################################
        power_spectrum = compute_power_spectrum_in_shanks(shank_signals, target_fs)
        ##### downsampled power spectrum
        downsampled_power_spectrum = downsample_spectrum_in_shanks(power_spectrum,1000)
        figure = plot_power_spectra(downsampled_power_spectrum)
        figure.savefig(figures_directory + 'power_spectrum_' + rat_names[rat_index] +
                       '_'+str(rat_sessions[rat_names[rat_index]][session_index]) + '.png')
        #################################################
        ##### combute ripple and theta bands total power
        low_freq = 100
        high_freq = 200
        ripple_power = compute_band_power_in_shanks(downsampled_power_spectrum,low_freq,high_freq)
        #low_freq = 6
        #high_freq = 12
        #theta_power = compute_band_power_in_shanks(downsampled_power_spectrum,low_freq,high_freq)
        spk_group = {}
        spk_group_probe_1 = channels['Probe1']['Spk.Group']
        spk_group_probe_2 = channels['Probe2']['Spk.Group']
        spk_group['Probe1'] = spk_group_probe_1
        spk_group['Probe2'] = spk_group_probe_2

        max_ripple_shanks = find_max_power_in_shanks(ripple_power)
        #max_theta_shanks = find_max_power_in_shanks(theta_power)

        combined_dict = {}
        for probe in spk_group:
            combined_dict[probe] = {}
            for shank in spk_group[probe]:
                combined_dict[probe][shank] = {
                    'spk.group': spk_group[probe][shank],
                    'max_ripple_power_channel': max_ripple_shanks[probe][shank],
                    #'max_theta_power_channel': max_theta_shanks[probe][shank]
                }

        #max_theta = find_max_value_and_index(theta_power)
        #combined_dict['max_theta'] = max_theta
        output_dict = {}
        for probe in spk_group:
            output_dict[probe] = {}
            for shank in spk_group[probe]:
                ripple_power_channel = combined_dict[probe][shank]['max_ripple_power_channel']
                #theta_power_channel = combined_dict[probe][shank]['max_theta_power_channel']
                output_dict[probe][shank] = {
                    'channel_information': combined_dict[probe][shank],
                    'ripple_channel': shank_signals[probe][shank][:,ripple_power_channel],
                    'power_spectrum':  downsampled_power_spectrum[probe][shank],
                }
            #max_theta_shank = combined_dict['max_theta'][probe]['shank']
            #max_theta_index = combined_dict['max_theta'][probe]['index']
            #theta_power_shank =  shank_signals[probe][max_theta_shank]
            #output_dict[probe]['theta'] = theta_power_shank[:,max_theta_index]

        import pickle as pkl
        # Define the filename where the dictionary will be stored
        output_filename = rat_names[rat_index]+'_' + rat_sessions[rat_names[rat_index]][session_index] + '_ripple_power_output.pkl'
        # Open the file for writing in binary mode and dump the dictionary
        with open(os.path.join(output_directory , output_filename), 'wb') as file:
            pkl.dump(output_dict, file)
        del power_spectrum
        del downsampled_power_spectrum
        del output_dict
        del combined_dict
        del shank_signals



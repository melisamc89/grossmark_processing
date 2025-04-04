Grossmark processing Pipeline

Overview
This repository contains a series of Python scripts used classification of neurons from Grossmark dataset (refer to https://crcns.org/data-sets/hc/hc-11/about-hc-11).


Data Set
This pipeline was developed to process the data set available from the CRCNS hc-11 data repository, which is associated with the paper titled "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." In this study, neural activity was recorded in CA1 pyramidal single units using two 8- or 6-shank silicon probes in the hippocampus of four rats. Recordings were taken during PRE, MAZE, and POST periods, with the maze alternating between a 1.6m linear wooden maze, a 2m linear metal maze, and a 1m diameter circular maze.

Scripts and Order of Execution
To ensure proper handling of data dependencies, the scripts should be executed in the following order:

0. Channel Information (channel_information.py)
Description: Defines the mapping of channels and shanks for different probes across multiple sessions and rats.
Here is also the information about rat names, session names and shanks and channels that are eventually discarded from the dataset.

Output: No direct file output but provides a crucial configuration for other scripts.

0. Configuration (config.py)
Description: Sets up general paths and directories used throughout the project. 
Should make sure that all directories are there (add checkin and if not create it)

Output: Establishes global variables for data, base, output, and figures directories.

1. LFP Analysis (lfp_analysis.py)
Input: EEG file (.eeg), session information (.mat)

Output: ripple_power_output.pkl - Contains channels with maximum ripple power during NREM periods, along with power spectra data.

2. LFP Theta Extraction (lfp_theta_extraction.py)
Input: Spike (.spk) and cluster (.clu) files.
Output: theta_output.pkl - Contains channels with maximum theta power during the maze period, along with extracted theta signals.

3. Spike Analysis (spk_analysis.py)
Input: Spike (.spk) and cluster (.clu) files.
Output: waveform_output.pkl - Contains mean waveform per cluster in each shank.

4. Deep Supervised Classification (deep_sup_classification.py)
Input: Outputs from lfp_analysis.py and spk_analysis.py.

Output: neuron_classification_output.pkl - Contains cell classification for each cluster found in each shank, including figures of waveforms and spectrograms.

5. Spikes Times Extraction (spikes_times_extraction.py)
Input: .res files, output from deep_sup_classification.py.

Output: stimes_classified.pkl - Contains a dictionary with lists of spike times and classifications.

6. Spikes Maze Analysis (spikes_maze_analysis.py)
Input: Outputs from spikes_times_extraction.py and contextual session information.

Output: neural_data.pkl - Contains spike matrix during the maze period.

7. Deep Sup Analysis
Input: Various outputs from the above steps.

Output: UMAP visualizations and further analyses.

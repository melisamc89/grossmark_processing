import copy
# Define the nested dictionary structure for the channel organization

number_of_channels = {
    'Achilles': 134,
    'Buddy': 132,
    'Cicero': 134,
    'Gatsby': 134
}

rat_names = ['Achilles','Buddy','Cicero','Gatsby']
rat_sessions = {}
rat_sessions['Achilles'] = ['10252013','11012013']
rat_sessions['Buddy'] = ['06272013']
rat_sessions['Cicero'] = ['09012014','09102014','09172014']
rat_sessions['Gatsby'] = ['08022013','08282013']

### will later create a loop on this
sessions = [[0,1],[0],[0,1,2],[0,1]]
#sessions = [[0],[0],[0],[0]]
#sessions = [[1],[0],[1,2],[1]]

low_ripple_freq = [100,100,100,100]
high_ripple_freq = [250,250,250,250]

channel_organization = {
    'Achilles': {},
    'Buddy': {},
    'Cicero': {},
    'Gatsby': {}
}
channel_organization['Achilles']['10252013'] = {
    'Probe1': {
        'LeftCA1Shank1': list(range(1, 11)),
        'LeftCA1Shank2': list(range(11, 21)),
        'LeftCA1Shank3': list(range(21, 31)),
        'LeftCA1Shank4': list(range(31, 41)),
        'LeftCA1Shank5': list(range(41, 51)),
        'LeftCA1Shank6': list(range(51, 61)),
        'LeftCA1Shank3Extra': list(range(61, 65)),
        'Spk.Group': {
            'LeftCA1Shank1': 1,
            'LeftCA1Shank2': 2,
            'LeftCA1Shank3': 3,
            'LeftCA1Shank4': 4,
            'LeftCA1Shank5': 5,
            'LeftCA1Shank6': 6,
            'LeftCA1Shank3Extra':7
        }#list(np.arange(1, 8))  # keep track of the spks files
    },
    'Probe2': {
        'RightCA1Shank1': list(range(65, 75)),
        'RightCA1Shank2': list(range(75, 85)),
        'RightCA1Shank3': list(range(85, 95)),
        'RightCA1Shank4': list(range(95, 105)),
        'RightCA1Shank5': list(range(105, 115)),
        'RightCA1Shank6': list(range(115, 125)),
        'RightCA1Shank7': list(range(125, 129)),
        'Spk.Group': {
            'RightCA1Shank1': 8,
            'RightCA1Shank2': 9,
            'RightCA1Shank3': 10,
            'RightCA1Shank4': 11,
            'RightCA1Shank5': 12,
            'RightCA1Shank6': 13,
            'RightCA1Shank7': 14,
        }#list(np.arange(8, 15))  #keep track of the spks files
    }
}
channel_organization['Achilles']['11012013'] = copy.deepcopy(channel_organization['Achilles']['10252013'])
##### HERE ADD THE CHANNEL ORGANIZATION FOR THE OTHER RATS
channel_organization['Buddy']['06272013'] = {
    'Probe1': {
        'LeftCA1Shank1': list(range(1, 9)),
        'LeftCA1Shank2': list(range(9, 17)),
        'LeftCA1Shank3': list(range(17, 24)),
        'LeftCA1Shank4': [25,26,28,29,30,31,32],
        #'LeftCA1Shank4': list(range(25, 33)),
        'LeftCA1Shank5': list(range(33, 41)),
        'LeftCA1Shank6': list(range(41, 49)),
        'LeftCA1Shank7': list(range(49, 56)), #### I will eliminate this shank
        'LeftCA1Shank8': [57,59,60,61,62,63,64],
        'Spk.Group': {
            'LeftCA1Shank1': 1,
            'LeftCA1Shank2': 2,
            'LeftCA1Shank3': 3,
            'LeftCA1Shank4': 4,
            'LeftCA1Shank5': 5,
            'LeftCA1Shank6': 6,
            'LeftCA1Shank7': 7,
            'LeftCA1Shank8': 8
        }  # list(np.arange(1, 8))  # keep track of the spks files
    },
    'Probe2': {
        'RightCA1Shank1': list(range(65, 73)),
        'RightCA1Shank2': list(range(73, 81)),
        'RightCA1Shank3': list(range(81, 89)),
        'RightCA1Shank4': list(range(89, 97)),
        'RightCA1Shank5': list(range(97, 105)),
        'RightCA1Shank6': list(range(105, 113)),
        'RightCA1Shank7': list(range(113, 120)), ### I will eliminate this shank
        'RightCA1Shank8': list(range(121, 129)),
        'Spk.Group': {
            'RightCA1Shank1': 9,
            'RightCA1Shank2': 10,
            'RightCA1Shank3': 11,
            'RightCA1Shank4': 12,
            'RightCA1Shank5': 13,
            'RightCA1Shank6': 14,
            'RightCA1Shank7': 15,
            'RightCA1Shank8': 16,
        }  # list(np.arange(8, 15))  #keep track of the spks files
    }
}
channel_organization['Cicero']['09012014'] = {
    'Probe1': {
        'LeftCA1Shank1': list(range(1, 11)),
        'LeftCA1Shank2': list(range(11, 21)),
        'LeftCA1Shank3': list(range(21, 31)),
        'LeftCA1Shank4': list(range(31, 41)),
        'LeftCA1Shank5': list(range(41, 51)),
        'LeftCA1Shank6': list(range(51, 61)),
        'LeftCA1Shank3Extra': list(range(61, 65)),
        'Spk.Group': {
            'LeftCA1Shank1': 1,
            'LeftCA1Shank2': 2,
            'LeftCA1Shank3': 3,
            'LeftCA1Shank4': 4,
            'LeftCA1Shank5': 5,
            'LeftCA1Shank6': 6,
            'LeftCA1Shank3Extra': 7
        }  # list(np.arange(1, 8))  # keep track of the spks files
    },
    'Probe2': {
        'RightCA1Shank1': list(range(65, 75)),
        'RightCA1Shank2': list(range(75, 85)),
        'RightCA1Shank3': list(range(85, 95)),
        'RightCA1Shank4': list(range(95, 105)),
        'RightCA1Shank5': list(range(105, 115)),
        'RightCA1Shank6': list(range(115, 125)),
        'RightCA1ExtraChannels': list(range(125, 129)),
        'Spk.Group': {
            'RightCA1Shank1': 8,
            'RightCA1Shank2': 9,
            'RightCA1Shank3': 10,
            'RightCA1Shank4': 11,
            'RightCA1Shank5': 12,
            'RightCA1Shank6': 13,
            'RightCA1ExtraChannels': 14,
        }  # list(np.arange(8, 15))  #keep track of the spks files
    }
}
channel_organization['Cicero']['09102014'] = copy.deepcopy(channel_organization['Cicero']['09012014'])
channel_organization['Cicero']['09172014'] = copy.deepcopy(channel_organization['Cicero']['09012014'])

channel_organization['Gatsby']['08022013'] = {
    'Probe1': {
        'LeftCA1Shank1': list(range(1, 9)),
        'LeftCA1Shank2': list(range(9, 17)),
        'LeftCA1Shank3': list(range(17, 25)),
        'LeftCA1Shank4': list(range(25, 33)),
        'LeftCA1Shank5': list(range(33, 41)),
        'LeftCA1Shank6': list(range(41, 49)),
        'LeftCA1Shank7': list(range(49, 57)),
        'LeftCA1Shank8': list(range(57, 64)), ### will eliminate this shank
        'Spk.Group': {
            'LeftCA1Shank1': 1,
            'LeftCA1Shank2': 2,
            'LeftCA1Shank3': 3,
            'LeftCA1Shank4': 4,
            'LeftCA1Shank5': 5,
            'LeftCA1Shank6': 6,
            'LeftCA1Shank7': 7,
            'LeftCA1Shank8': 8,
        }  # list(np.arange(1, 8))  # keep track of the spks files
    },
    'Probe2': {
        'RightCA1Shank1': list(range(65, 73)),
        'RightCA1Shank2': list(range(73, 81)),
        'RightCA1Shank3': list(range(81, 89)),
        'RightCA1Shank4': list(range(89, 97)),
        'RightCA1Shank5': list(range(97, 105)),
        'RightCA1Shank6': list(range(105, 113)),
        'RightCA1Shank7': list(range(113, 121)),
        'RightCA1Shank8': list(range(121, 129)),
        'Spk.Group': {
            'RightCA1Shank1': 9,
            'RightCA1Shank2': 10,
            'RightCA1Shank3': 11,
            'RightCA1Shank4': 12,
            'RightCA1Shank5': 13,
            'RightCA1Shank6': 14,
            'RightCA1Shank7': 15,
            'RightCA1Shank8': 16,
        }  # list(np.arange(8, 15))  #keep track of the spks files
    }
}
channel_organization['Gatsby']['08282013'] = {
    'Probe1': {
        'LeftCA1Shank1': list(range(1, 9)),
        'LeftCA1Shank2': list(range(9, 17)),
        'LeftCA1Shank3': list(range(17, 25)),
        'LeftCA1Shank4': list(range(25, 33)),
        'LeftCA1Shank5': [33,34,36,37,38,39,40],
        'LeftCA1Shank6': [41,42,43,44,46,48],
        'LeftCA1Shank7': list(range(49, 57)),
        'LeftCA1Shank8': list(range(57, 65)),
        'Spk.Group': {
            'LeftCA1Shank1': 1,
            'LeftCA1Shank2': 2,
            'LeftCA1Shank3': 3,
            'LeftCA1Shank4': 4,
            'LeftCA1Shank5': 5,
            'LeftCA1Shank6': 6,
            'LeftCA1Shank7': 7,
            'LeftCA1Shank8': 8,
        }  # list(np.arange(1, 8))  # keep track of the spks files
    },
    'Probe2': {
        'RightCA1Shank1': list(range(65, 73)),
        'RightCA1Shank2': list(range(73, 81)),
        'RightCA1Shank3': list(range(81, 89)),
        'RightCA1Shank4': list(range(89, 97)),
        'RightCA1Shank5': list(range(97, 105)),
        'RightCA1Shank6': list(range(105, 113)),
        'RightCA1Shank7': list(range(113, 121)),
        'RightCA1Shank8': list(range(121, 129)),
        'Spk.Group': {
            'RightCA1Shank1': 9,
            'RightCA1Shank2': 10,
            'RightCA1Shank3': 11,
            'RightCA1Shank4': 12,
            'RightCA1Shank5': 13,
            'RightCA1Shank6': 14,
            'RightCA1Shank7': 15,
            'RightCA1Shank8': 16,
        }  # list(np.arange(8, 15))  #keep track of the spks files
    }
}

discarded_shanks = {}
discarded_shanks['Achilles'] = {}
discarded_shanks['Achilles']['10252013'] = {}
discarded_shanks['Achilles']['10252013']['Probe1'] = ['LeftCA1Shank6','LeftCA1Shank3Extra']
discarded_shanks['Achilles']['10252013']['Probe2'] = []
discarded_shanks['Achilles']['11012013'] = {}
discarded_shanks['Achilles']['11012013']['Probe1'] = []
discarded_shanks['Achilles']['11012013']['Probe2'] = []
discarded_shanks['Buddy'] = {}
discarded_shanks['Buddy']['06272013'] = {}
discarded_shanks['Buddy']['06272013']['Probe1'] = ['LeftCA1Shank7']
discarded_shanks['Buddy']['06272013']['Probe2'] = ['RightCA1Shank7']
discarded_shanks['Cicero'] = {}
discarded_shanks['Cicero']['09012014'] = {}
#discarded_shanks['Cicero']['09012014']['Probe1'] = ['LeftCA1Shank3Extra']
discarded_shanks['Cicero']['09012014']['Probe1'] = []
#discarded_shanks['Cicero']['09012014']['Probe2'] = ['RightCA1ExtraChannels']
discarded_shanks['Cicero']['09012014']['Probe2'] = []
discarded_shanks['Cicero']['09102014'] = {}
#discarded_shanks['Cicero']['09102014']['Probe1'] = ['LeftCA1Shank3Extra']
discarded_shanks['Cicero']['09102014']['Probe1'] = []
#discarded_shanks['Cicero']['09102014']['Probe2'] = ['RightCA1ExtraChannels']
discarded_shanks['Cicero']['09102014']['Probe2'] = []
discarded_shanks['Cicero']['09172014'] = {}
discarded_shanks['Cicero']['09172014']['Probe1'] = ['LeftCA1Shank3Extra']
discarded_shanks['Cicero']['09172014']['Probe1'] = []
discarded_shanks['Cicero']['09172014']['Probe2'] = ['RightCA1ExtraChannels']
discarded_shanks['Cicero']['09172014']['Probe2'] = []

discarded_shanks['Gatsby'] = {}
discarded_shanks['Gatsby']['08022013'] = {}
#discarded_shanks['Gatsby']['08022013']['Probe1'] = ['LeftCA1Shank8']
discarded_shanks['Gatsby']['08022013']['Probe1'] = []
discarded_shanks['Gatsby']['08022013']['Probe2'] = []

discarded_shanks['Gatsby']['08282013'] = {}
discarded_shanks['Gatsby']['08282013']['Probe1'] = []
discarded_shanks['Gatsby']['08282013']['Probe2'] = []




import pandas as pd
import numpy as np

spikes_matrix = output_dic['spikes_matrix']
layerID = output_dic['LayerID']

# Find indices where the value is 'DEEP'
deep_index = np.array([index for index, value in enumerate(layerID ) if value == 'DEEP'])
# Find indices where the value is 'SUPERFICIAL'
superficial_index = np.array([index for index, value in enumerate(layerID ) if value == 'SUPERFICIAL'])

deep_spikes = spikes_matrix[:,deep_index]
sup_spikes =  spikes_matrix[:,superficial_index]





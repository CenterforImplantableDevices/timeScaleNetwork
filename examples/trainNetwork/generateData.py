from timeScaleNetwork import noise as tisc_noise
from fcn_plotEmbedding import make_square
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import itertools
import torch
import wave
import math
import os



# Define Scales containing target waveforms
scales_include = [12,7,10,8,9]
index_partialStart = 1

# Define length of the signal
length_signal_exp = 16
scale_multiplier  = 2

# Define Outputs
show_plot = False
path_save = './data_synthetic'
print(os.getcwd())
if not( os.path.exists(path_save)) and not( os.path.isdir(path_save)):
    os.mkdir(path_save)



'''
    Define Data
'''
# Create template waveforms
time = torch.linspace(0, 2*math.pi, scale_multiplier**length_signal_exp)
template0 = torch.sin( time).tolist()                   # Sin Wave
template1 = torch.cos( time).tolist()                   # Cos Wave
template2 = signal.sawtooth(time).tolist()              # Sawtooth Wave
template3 = signal.morlet(len(time)).real
template3 = (template3 / template3.max()).tolist()      # Morlet Wavelet
template4 = signal.ricker(len(time),len(time)//6).real
template4 = (template4 / template4.max()).tolist()      # Mexican Hat wavelet
templates   = [template0, template1, template2, template3, template4]
# Generate all combinations of all scales
for l in range(1,len(scales_include)+1):
    for subscales in itertools.combinations(scales_include, l):
        index_subscales = [scales_include.index(s) for s in subscales]

        '''
            Define Data
        '''
        # Define dilation
        dilation_exponent = [length_signal_exp - target for target in subscales]
        # Define the data components that will create the final data
        data_components = [templates[i][:: scale_multiplier ** exp] * (scale_multiplier**(exp-1))  for i, exp in zip( index_subscales, dilation_exponent)]
        # Calculate utility parameters
        len_whole   = len(data_components[0])
        noise = tisc_noise.get_white(time.shape)
        noise = (noise / noise.max()).tolist()
        # Combine all data components to one data_target tensor
        data_components = data_components + [noise[:len_whole]]
        data_target     = torch.sum( torch.Tensor( data_components), dim=0)

        '''
            Define Label
        '''
        label = torch.zeros(len(templates))
        for i in index_subscales:
            label[i] = 1
        print('Label: ', label)

        '''
            Define Outputs
        '''
        if show_plot:
            plt.figure()
            plt.plot(data_target.detach().numpy())
            plt.title('Data - ' + ','.join([str(s) for s in subscales]))
            plt.show()
        
        filename = 'segments_' + '-'.join([str(s) for s in subscales])
        data_save = data_target.numpy()
        with wave.open( os.path.join( path_save, filename+'_data.wav'), 'w') as f:
            f.setparams((1, data_save.itemsize, 1, 0, 'NONE','not compressed'))
            f.writeframes( data_save.tobytes())
        label_save = label.numpy()
        with wave.open( os.path.join( path_save, filename+'_label.wav'), 'w') as f:
            f.setparams((1, label_save.itemsize, 1, 0, 'NONE','not compressed'))
            f.writeframes( label_save.tobytes())

# Lastly, add an empty noise sample
noise = tisc_noise.get_white(time.shape)
data_save = torch.Tensor(noise / noise.max()).numpy()[:len_whole]
with wave.open( os.path.join( path_save, 'segments_None_data.wav'), 'w') as f:
    f.setparams((1, data_save.itemsize, 1, 0, 'NONE','not compressed'))
    f.writeframes( data_save.tobytes())
label_save = torch.zeros(len(templates)).numpy()
with wave.open( os.path.join( path_save, 'segments_None_label.wav'), 'w') as f:
    f.setparams((1, label_save.itemsize, 1, 0, 'NONE','not compressed'))
    f.writeframes( label_save.tobytes())

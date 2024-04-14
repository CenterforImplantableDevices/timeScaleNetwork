from timeScaleNetwork import noise as tisc_noise
from fcn_plotEmbedding import make_square
import matplotlib.pyplot as plt
from scipy import signal
import torch
import math
import wave
import os



# Define Scales containing target waveforms
scale_target_whole = [12]
scale_target_partial = [7,10,8,9]

# Define the length of the signal
scale_multiplier  = 2
length_signal_exp = 16

# Define Outputs
show_plot = True
save_plot = True
save_data = True
path_save = './synthetic_output'
if not( os.path.exists(path_save)) and not( os.path.isdir(path_save)):
    os.mkdir(path_save)



'''
    Define Data
'''
# Create template waveforms
time = torch.linspace(0, 2*math.pi, scale_multiplier**length_signal_exp)
noise_whole    = tisc_noise.get_white(time.shape)
noise_whole    = (noise_whole / noise_whole.max()).tolist()
template_whole = torch.sin( time).tolist()
template0 = torch.cos( time).tolist()                   # Cos Wave
template1 = signal.sawtooth(time).tolist()              # Sawtooth Wave
template2 = signal.morlet(len(time)).real
template2 = (template2 / template2.max()).tolist()      # Morlet Wavelet
template3 = signal.ricker(len(time),len(time)//6).real
template3 = (template3 / template3.max()).tolist()      # Mexican Hat wavelet
template_partial   = [template0, template1, template2, template3]
# Define dilation
dilation_exponent_whole   = [length_signal_exp - target for target in scale_target_whole]
dilation_exponent_partial = [length_signal_exp - target if target is not None else None for target in scale_target_partial]
# Define the data components that will create the final data
data_target_whole   = [template_whole[     :: scale_multiplier ** exp] * (scale_multiplier**(exp-1)) for i, exp in enumerate(dilation_exponent_whole)]
data_target_partial = [template_partial[i][:: scale_multiplier ** exp] * (scale_multiplier**(exp-1)) if exp is not None else None for i, exp in enumerate(dilation_exponent_partial)]
# Calculate utility parameters
len_whole   = len(data_target_whole[0])
len_partial = int( len_whole / len(data_target_partial))
# Combine the partial segments into one list
data_target_partialALL = []
for data in data_target_partial:
    if data is None:
        data_target_partialALL.extend([0] * len_partial)
    else:
        data_target_partialALL.extend(data[:len_partial])
# Combine all data components to one data_target tensor
data_target = data_target_whole + [data_target_partialALL] + [noise_whole[:len_whole]]
data_target = torch.sum( torch.Tensor( data_target), dim=0)

'''
    Define Label
'''
# Define the indexes for each scale that was included in data (entire frame)
index_target_label = []
for scale in scale_target_whole:
    # Define scale information
    len_scale = scale_multiplier ** (length_signal_exp - scale -1)
    # Use this to define start and ending indexes
    index_start = len_scale -1
    index_end   = index_start * scale_multiplier +1
    index_target_label.append([index_start,index_end])
# Define the indexes for each scale that was included in data (partial frame)
fraction_partial = len_partial / len_whole
for i, scale in enumerate(scale_target_partial):
    if scale is None:
        index_target_label.append([None,None])
    else:
        # Calculate scale information
        len_scale = scale_multiplier ** (length_signal_exp - scale -1)
        len_partial = int(len_scale * fraction_partial)
        # Use this to define start and ending indexes
        index_start = len_scale -1 + i * len_partial
        index_end   = index_start + len_partial
        index_target_label.append([index_start,index_end])
# Combine all scale components to one scale_target list tensor
scale_target = scale_target_whole + scale_target_partial
# Define the label array
label_target = torch.zeros(len_whole -1)
for (start, stop), scale in zip(index_target_label, scale_target):
    if scale is None:
        continue
    else:
        label_target[start:stop] = scale_multiplier ** scale

'''
    Define Outputs
'''
filename = 'combo_' + '-'.join([str(s) for s in scale_target_whole]) + '_' + '-'.join([str(s) for s in scale_target_partial])
if show_plot:
    linewidth = 3
    path_savePlot = os.path.join( path_save, 'plots')
    if save_plot and not( os.path.exists(path_savePlot)) and not( os.path.isdir(path_savePlot)):
        os.mkdir(path_savePlot)

    plt.figure()
    plt.plot(template_whole, 'k', linewidth=linewidth)
    plt.title('Template Whole')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_templateWhole.png'))

    plt.figure()
    plt.plot(template0, 'k', linewidth=linewidth)
    plt.title('Template 0')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_template0.png'))

    plt.figure()
    plt.plot(template1, 'k', linewidth=linewidth)
    plt.title('Template 1')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_template1.png'))

    plt.figure()
    plt.plot(template2, 'k', linewidth=linewidth)
    plt.title('Template 2')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_template2.png'))

    plt.figure()
    plt.plot(template3, 'k', linewidth=linewidth)
    plt.title('Template 3')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_template3.png'))

    plt.figure()
    plt.plot(data_target.detach().numpy(), 'k', linewidth=1)
    plt.title('Data')
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_data.png'))

    plt.figure()
    label_to_plot = make_square( label_target, normalize=True).numpy()
    plt.imshow( label_to_plot, aspect='auto')
    # plt.colorbar()
    plt.title('Labels')
    plt.ylabel('Scale')
    plt.xlabel('Data Index')
    plt.xticks([],[])
    # plt.tick_params( axis='x',which='both', bottom=False, top=False, labelbottom=True) 
    plt.show()
    if save_plot: plt.savefig( os.path.join( path_savePlot, filename+'_label.png'))

if save_data:
    data_save = data_target.numpy()
    with wave.open( os.path.join( path_save, filename+'_data.wav'), 'w') as f:
        f.setparams((1, data_save.itemsize, 1, 0, 'NONE','not compressed'))
        f.writeframes( data_save.tobytes())

    label_save = label_target.numpy()
    with wave.open( os.path.join( path_save, filename+'_label.wav'), 'w') as f:
        f.setparams((1, label_save.itemsize, 1, 0, 'NONE','not compressed'))
        f.writeframes( label_save.tobytes())

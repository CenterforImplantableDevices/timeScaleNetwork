from examples.trainLayer.fcn_plotEmbedding import make_square, make_exponential_list
from timeScaleNetwork import timescaleL
import matplotlib.pyplot as plt
import numpy as np
import torch
import wave
import os



# Define Training Parameters
num_epochs = 50
TiScLayer  = timescaleL.TSC_input(  2, 16384, initialization='white')
loss_fn    = torch.nn.MSELoss()
optimizer     = 'sgd'
learning_rate = 1e-2
weight_decay  = 1
momentum      = 0

# Define Outputs
save_output   = False
save_weights  = False
save_gradient = False
path_data     = './synthetic_output/combo_12_7-10-8-9'
path_save     = './synthetic_output/training_output'
if not( os.path.exists(path_save)) and not( os.path.isdir(path_save)):
    os.mkdir(path_save)



'''
    Setup Training
'''
# Load Data
with wave.open( path_data+'_data.wav', 'rb') as f:
    data  = np.frombuffer( f.readframes( f.getnframes()), dtype=np.float32)
    data  = torch.Tensor(data.copy())
# Load Label
with wave.open( path_data+'_label.wav', 'rb') as f:
    label = np.frombuffer( f.readframes( f.getnframes()), dtype=np.float32)
    label = torch.Tensor(label.copy())
# Initialize Optimizer
if optimizer.lower() == 'sgd':
    optimizer_fn = torch.optim.SGD(      filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer.lower() == 'rmsprop':
    optimizer_fn = torch.optim.RMSprop(  filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer.lower() == 'adam':
    optimizer_fn = torch.optim.Adam(     filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adagrad':
    optimizer_fn = torch.optim.Adagrad(  filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'asgd':
    optimizer_fn = torch.optim.ASGD(     filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'rprop':
    optimizer_fn = torch.optim.Rprop(    filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate)
elif optimizer.lower() == 'adadelta':
    optimizer_fn = torch.optim.Adadelta( filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adamw':
    optimizer_fn = torch.optim.AdamW(    filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adamax':
    optimizer_fn = torch.optim.Adamax(   filter(lambda p: p.requires_grad, TiScLayer.parameters()), lr=learning_rate, weight_decay=weight_decay)
# Misc Setup
TiScLayer.train()
data  = data.unsqueeze(0).requires_grad_()
label = label.unsqueeze(0)



''' 
    Train Layer
'''
for epoch in range( num_epochs):
    # Implement on epoch
    optimizer_fn.zero_grad()
    out  = TiScLayer.forward(data)
    loss = loss_fn( out, label)
    loss.backward()

    # Save outputs
    if save_output:
        plt.figure()
        data_to_plot = make_square( out.squeeze(), normalize=True).detach().numpy()
        plt.imshow( data_to_plot, aspect='auto')
        plt.colorbar()
        plt.title('Output')
        plt.savefig( os.path.join(path_save, 'output_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

    if save_weights:
        plt.figure()
        data_to_plot = TiScLayer.bias.clone().detach().unsqueeze(0).transpose(0,1).numpy()[::-1]
        plt.imshow( data_to_plot, aspect='auto')
        plt.colorbar()
        plt.title('WEIGHTS BIAS')
        plt.savefig( os.path.join(path_save, 'bias_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

        plt.figure(figsize=[8,16])
        data_to_plot = make_exponential_list( TiScLayer.weights.clone().detach().squeeze())
        num_weights = len( data_to_plot)
        for i, wgt in enumerate( data_to_plot):
            plt.subplot( num_weights, 1, i+1)
            plt.plot( wgt[0])
        plt.subplot(num_weights, 1, 1)
        plt.title('WEIGHTS')
        plt.savefig( os.path.join(path_save, 'weights_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

    if save_gradient:
        plt.figure()
        data_to_plot = TiScLayer.bias.grad.clone().detach().unsqueeze(0).transpose(0,1).numpy()[::-1]
        plt.imshow( data_to_plot, aspect='auto')
        plt.colorbar()
        plt.title('WEIGHTS BIAS')
        plt.savefig( os.path.join(path_save, 'biasGrad_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

        plt.figure(figsize=[8,16])
        data_to_plot = make_exponential_list( TiScLayer.weights.grad.clone().detach().squeeze())
        num_weights = len( data_to_plot)
        for i, wgt in enumerate( data_to_plot):
            plt.subplot( num_weights, 1, i+1)
            plt.plot( wgt[0])
        plt.subplot(num_weights, 1, 1)
        plt.title('WEIGHTS')
        plt.savefig( os.path.join(path_save, 'weightsGrad_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

    optimizer_fn.step()
    print('\tEpoch ', epoch, '\tLoss: ', loss.item())


plt.figure()
data_to_plot = make_exponential_list( TiScLayer.weights.clone().detach().squeeze())[1:]
num_weights = len( data_to_plot)
for i, wgt in enumerate( data_to_plot):
    plt.subplot( num_weights, 1, i+1)
    plt.plot( wgt[0], 'k', linewidth=2)
    plt.ylabel(i)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.box(False)
plt.suptitle('TRAINED WEIGHTS')
plt.show()
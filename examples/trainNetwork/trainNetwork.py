from fcn_plotEmbedding import make_exponential_list
from timeScaleNetwork import timescaleN
import matplotlib.pyplot as plt
import numpy as np
import torch
import wave
import os



# Define Training Parameters
num_epochs  = 50
TiScNetwork = timescaleN.TiscMlpN( [[2, 2**15], 5], length_input=32768, tisc_initialization='white')
loss_fn     = torch.nn.MSELoss() # For Classification, try CrossEntropy loss functions
optimizer     = 'sgd'            # For Classificaiton, try RMSprop or ADAM optimizers
learning_rate = 1e-2
weight_decay  = 1
momentum      = 0

# Define Outputs
save_weights  = True
path_data     = './data_synthetic'
path_save     = './output_training'
if not( os.path.exists(path_data)) and not( os.path.isdir(path_data)):
    print('No Data Found. You may need to run generateData.py first.')
    import sys
    sys.exit()
if not( os.path.exists(path_save)) and not( os.path.isdir(path_save)):
    os.mkdir(path_save)



'''
    Setup Training
'''
# Initialize Optimizer
if optimizer.lower() == 'sgd':
    optimizer_fn = torch.optim.SGD(      filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer.lower() == 'rmsprop':
    optimizer_fn = torch.optim.RMSprop(  filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer.lower() == 'adam':
    optimizer_fn = torch.optim.Adam(     filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adagrad':
    optimizer_fn = torch.optim.Adagrad(  filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'asgd':
    optimizer_fn = torch.optim.ASGD(     filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'rprop':
    optimizer_fn = torch.optim.Rprop(    filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate)
elif optimizer.lower() == 'adadelta':
    optimizer_fn = torch.optim.Adadelta( filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adamw':
    optimizer_fn = torch.optim.AdamW(    filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
elif optimizer.lower() == 'adamax':
    optimizer_fn = torch.optim.Adamax(   filter(lambda p: p.requires_grad, TiScNetwork.parameters()), lr=learning_rate, weight_decay=weight_decay)
TiScNetwork.train()
# Setup Data
class dset(torch.utils.data.Dataset):
    '''Class to define dataset that will be iterated over during training.
    '''
    def __init__(self, path_data):
        samples = []
        self.data = []
        for file in os.listdir(path_data):
            if not file.endswith('.wav'):
                continue
            # Strip the last descriptor off (will be either data or label)
            sample_desc = '_'.join(file.split('_')[:-1])

            # If the sample is new, add it to self.data
            if sample_desc not in samples:
                samples.append(sample_desc)
                with wave.open( os.path.join( path_data, sample_desc+'_data.wav'), 'rb') as f:
                    sample  = np.frombuffer( f.readframes( f.getnframes()), dtype=np.float32)
                    sample = torch.Tensor(sample.copy())
                with wave.open( os.path.join( path_data, sample_desc+'_label.wav'), 'rb') as f:
                    label = np.frombuffer( f.readframes( f.getnframes()), dtype=np.float32)
                    label = torch.Tensor(label.copy())
                self.data.append([sample,label])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        d = self.data[idx]
        return d[0], d[1]

synthetic_dset = dset(path_data)
dataloader = torch.utils.data.DataLoader( synthetic_dset, batch_size=1, shuffle=True, num_workers=0)



'''
    Train Network
'''
for epoch in range( num_epochs):
    # Implement one epoch
    for sample, label in dataloader:
        optimizer_fn.zero_grad()
        out  = TiScNetwork.forward(sample)
        loss = loss_fn( out, label)
        loss.backward()
        optimizer_fn.step()

    # Save outputs
    if save_weights:
        plt.figure()
        data_to_plot = TiScNetwork.layers[0][0][0].bias.clone().detach().unsqueeze(0).transpose(0,1).numpy()[::-1]
        plt.imshow( data_to_plot, aspect='auto')
        plt.colorbar()
        plt.title('WEIGHTS BIAS')
        plt.savefig( os.path.join(path_save, 'bias_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

        plt.figure(figsize=[8,16])
        data_to_plot = make_exponential_list( TiScNetwork.layers[0][0][0].weights.clone().detach().squeeze())
        num_weights = len( data_to_plot)
        for i, wgt in enumerate( data_to_plot):
            plt.subplot( num_weights, 1, i+1)
            plt.plot( wgt[0])
        plt.subplot(num_weights, 1, 1)
        plt.title('WEIGHTS')
        plt.savefig( os.path.join(path_save, 'weights_epoch-'+str(epoch)+'.png'))
        plt.clf()
        plt.close()

    print('\tEpoch ', epoch, '\tLoss: ', loss.item())


TiScNetwork.eval()
correct = 0
for sample, label in dataloader:
    out  = TiScNetwork.forward(sample)
    label_list  = [ l > 0.5 for l in label.clone().detach().squeeze().tolist()]
    output_list = [ p > 0.5 for p in out.clone().detach().squeeze().tolist()]
    # print('\nl: ',  label_list)
    # print('o: ', output_list)

    if not( all( [l == p for l,p in zip( label_list, output_list)])):
        print('\nl: ',  label_list)
        print('o: ', output_list)
        print('INCORRECT')
    else:
        correct += 1
print('Accuracy: {:.2f}%'.format(100*(correct/len(dataloader))))

plt.figure()
data_to_plot = make_exponential_list( TiScNetwork.layers[0][0][0].weights.clone().detach().squeeze())[1:]
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
from timeScaleNetwork import timescaleN as tiscNetwork
from timeScaleNetwork import timescaleL as tiscLayer
from numpy.random import rand
import torch



# Note, it's best if all channel dimensions are a power of 2 (2, 4, 8, 16, ...)
# Excess length beyond the closest power will be padded/clipped

# Sample 2-channel data, with individual channels on each row
data_channelRow = rand(2, 16390)
# Flatten the data before pssing into the network
data_flattened = data_channelRow.flatten(order='F')
tensor_flattened = torch.Tensor(data_channelRow).t().reshape(-1)

# Sample 2-channel data, with individual channels on each column
data_channelColumn = rand(16384, 2)
# Flatten the data before pssing into the network
data_flattened = data_channelColumn.flatten(order='C')
tensor_flattened = torch.Tensor(data_channelColumn).reshape(-1)
print('Data Raw:\n', data_channelColumn)
print('Data Flattened:\n', tensor_flattened)


# Create a 1-layer TiSc Network
#   (input) minimum window = length 2
#   (input) maximum window = lenght 16384
#   Dense Network classifier with hidden layers size 300, 100, 2 (one-hot output)
testNet_1layer = tiscNetwork.TiscMlpN( [[2, 16384], 300, 100, 2], num_tisc_channels=2, length_input=data_flattened.shape[0], dropout=[True, True, False])
print('\n---\n\nTiSc Input Layer (2 Independent Channels)')
print(testNet_1layer)
output = testNet_1layer( tensor_flattened)
print('Output:')
print(output)

# Create a 2-layer TiSc Network
#   (input)  minimum window = length 2
#   (input)  maximum window = lenght 16384
#   (hidden) minimum window = length
#   (hidden) maximum window = length
#   Dense Network classifier with hidden layers size 300, 100, 2 (one-hot output)
# Create a 2-layer TiSc Network
testNet_2layer = tiscNetwork.TiscMlpN( [[2, 16384], [2, 16384, False], 300, 100, 2], num_tisc_channels=2, length_input=data_flattened.shape[0], dropout=[True, True, False])
print('\n---\n\nTiSc Input Layer and TiSc Hidden Layer (2 Independent Channels)')
print(testNet_1layer)
output = testNet_1layer( tensor_flattened)
print('Output')
print(output)



# Create a single layer, for integration into a larger network
testLayer_input  = tiscLayer.TSC_input( 2, 16384)
embedding_1 = testLayer_input( tensor_flattened)

# output_size = ((length_input // tisc_temp.min_window) * 2 -1)
output_size = embedding_1.shape[-1]
testLayer_hidden = tiscLayer.TSC_hidden( 2, 16384, output_size, False)
embedding_2 = testLayer_hidden( embedding_1)
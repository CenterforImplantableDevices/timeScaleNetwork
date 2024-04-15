from timeScaleNetwork import timescaleN as tiscNetwork
from timeScaleNetwork import timescaleL as tiscLayer
from numpy.random import rand
import torch



# It's best if all dimensions are a power of 2 (2, 4, 8, 16, ...)
# Excess values beyond the closest power will be automatically padded/clipped
# Comments labeled "DIMS" help track data dimensions



# Sample 2-channel data, with individual channels on each row
data_channelRow = rand(2, 16390)
# Flatten the data before passing into the network
data_flattened = data_channelRow.flatten(order='F')
tensor_flattened = torch.Tensor(data_channelRow).t().reshape(-1)

# Sample 2-channel data, with individual channels on each column
data_channelColumn = rand(16384, 2)
# Flatten the data before pssing into the network
# DIMS: 16384 * 2 = 32768
data_flattened = data_channelColumn.flatten(order='C')
tensor_flattened = torch.Tensor(data_channelColumn).reshape(-1)



# Create a 1-layer TiSc Network
#   (input) minimum window = length 2
#   (input) maximum window = length 16384
#   DIMS max:  32768 -> (min window 2) -> 16384
#   DIMS total: (16384 *2-1) * 2 channels = 65534
#   Dense Network classifier with hidden layers size 300, 100, 2 (one-hot output)
testNet_1layer = tiscNetwork.TiscMlpN( [[2, 16384], 300, 100, 2], num_tisc_channels=2, length_input=data_flattened.shape[0], dropout=[True])
output_1layer = testNet_1layer( tensor_flattened)

# Create a 3-layer TiSc Network
#   (input)  minimum window = length 2
#   (input)  maximum window = length 16384
#   DIMS max:  32768 -> (min window 2) -> 16384
#   (hidden) minimum window = length 4
#   (hidden) maximum window = length 16384
#   DIMS max:  16384 -> (min window 4) -> 4096
#   (hidden) minimum window = length 2
#   (hidden) maximum window = length 4096
#   DIMS max:  4096 -> (min window 2) -> 2048
#   DIMS total: (2048 *2-1) * 3 channels = 12285
#   Dense Network classifier with hidden layers size 300, 100, 2 (one-hot output)
# Create a 2-layer TiSc Network
testNet_3layer = tiscNetwork.TiscMlpN( [[2, 16384], [4, 16384], [2, 4096], 300, 100, 2], num_tisc_channels=3, length_input=data_flattened.shape[0], dropout=[True, True, False])
output_3layer = testNet_3layer( tensor_flattened)



# Create a single layer, for integration into a larger network
testLayer_input  = tiscLayer.TSC_input( 2, 16384)
embedding_1 = testLayer_input( tensor_flattened)

# output_size = ((length_input // testLayer_input.min_window) * 2 -1)
output_size = embedding_1.shape[-1]
testLayer_hidden = tiscLayer.TSC_hidden( 2, 16384, output_size, False)
embedding_2 = testLayer_hidden( embedding_1)



print('Data Raw:\n', data_channelColumn)
print('Data Flattened:\n', tensor_flattened)

print('\n---\n\nTiSc Input Layer Only (2 Independent Channels)\n', testNet_1layer)
print('Output:\n', output_1layer)

print('\n---\n\nTiSc Input Layer and TiSc Hidden Layer (2 Independent Channels)\n', testNet_3layer)
print('Output:\n', output_3layer)
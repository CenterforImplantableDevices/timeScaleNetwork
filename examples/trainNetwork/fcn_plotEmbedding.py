import torch
import math


def make_square( input_data, data_dim=0, scale_multiplieriplier=2, normalize=True):
    '''Separates TiSc data into a 2D array with each scale in one row, with each scale stretched to make an even length 2D array.

    In this output each column index correlates wtih a unit of time.

    Args:
        input_data (numpy array):         TiSc organized data
        data_dim (int, optional):         Dimension along which the TiSc organized data is organized.
        scale_multiplieriplier (int, optional): Exponential base designating how fast the window size increases.
        normalize (bool, optional):       Whether to independently normalize each scale to between 0 and 1.

    Returns:
        Numpy ndarray: 2D arra where rows are each scale the the columns are each time index.
    '''
    # Get the starting index for the smallest scale (smallest window, most values), and the number of levels to iterate for
    start_index = int( (input_data.shape[data_dim]+1) / scale_multiplieriplier)
    num_levels  = int( math.log( input_data.shape[data_dim], scale_multiplieriplier))+1

    # Create the output array
    output = torch.zeros((num_levels, start_index))
    num_repeats = 1
    # Iterate thorugh each scale and insert the associated scale values into the proper index in the output array.
    for i in range( num_levels):
        # Expand the length of the data by using the repeat_interleave function
        output[i, :] = input_data[ start_index-1 : start_index*scale_multiplieriplier-1].repeat_interleave( num_repeats)
        # If we normalize each scale, divide by num_repeats
        # wich increases as the scale gets larger (i.e. window size and output values get bigger)
        if normalize:
            output[i, :] = output[i, :].div(num_repeats)

        # Increment the indexing values
        start_index  = int( start_index / scale_multiplieriplier)
        num_repeats *= scale_multiplieriplier
    
    return output



def make_exponential_list( input_data, data_dim=0, scale_multiplier=2, normalize=True):
    '''Separates TiSc data into a 1D python list of 1D Numpy arrays were each scale is a list element. Each scale remains its original length.

    Args:
        input_data (numpy array):         TiSc organized data
        data_dim (int, optional):         Dimension along which the TiSc organized data is organized.
        scale_multiplier (int, optional): Exponential base designating how fast the window size increases.
        normalize (bool, optional):       Whether to independently normalize each scale to between 0 and 1.

    Returns:
        List where each element is the scale, containing a 1D numpy array where the columns are the scale index.
    '''
    # Get the starting index for the smallest scale (smallest window, most values), and the number of levels to iterate for
    start_index = int( (input_data.shape[data_dim]+1) / scale_multiplier)
    num_levels  = int( math.log( input_data.shape[data_dim], scale_multiplier))+1

    # Create the output array
    output = []
    num_repeats = 1
    # Iterate thorugh each scale and append the associated scale values to the proper index in the output array.
    for i in range( num_levels-1, -1, -1):
        out = input_data[ start_index-1 : start_index*scale_multiplier-1]
        # If we normalize each scale, divide by num_repeats
        # wich increases as the scale gets larger (i.e. window size and output values get bigger)
        if normalize:
            out = out.div(num_repeats)
        
        output.append(out.detach().unsqueeze(0).numpy())

        # Increment the indexing values
        start_index  = int( start_index / scale_multiplier)
        num_repeats *= scale_multiplier
    
    # Since we built the output array backwards, reverse it before returning
    return output[::-1]
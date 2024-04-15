'''
    A Class that contains the low-level operations performed by a TiSc Input Layer.
'''
import torch



class TiScFunction_input(torch.autograd.Function):
    ''' Class defining static forward() and backward() methods for use by :py:class:`~toast.heat.timescaleL.TSC_input`
    '''

    @staticmethod
    def forward(ctx, data, weights, bias, min_window_exponential, max_window_exponential, scale_multiplier=2):
        '''forward method defining the forward pass operations.

        Bias must be included, but its value may be ``None``

        Important:
            Assumes input data is of dim=2 where dimensions represent (data_size, batch_size)

        Note:
            It is up to the user to verfify that the lengths of data samples (``data.shape[0]``) is an even multiple of the max_window size.
            Invalid dimension errors during windowing/reshaping commands will be raised if this is not true.

        Args:
            data (Tensor):                    1D Tensor holding the input data.
            weights (Tensor):                 1D Tensor holding the weights. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):                    1D Tensor holding the bias values.
            min_window_exponential (int):     Exponent value designating the minimum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            max_window_exponential (int):     Exponent value designating the maximum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            scale_multiplier (int, optional): Exponential base designating how fast the window size increases.
        
        Returns:
            1D Tensor representing the output. Note these values should be interpreted in a pseudo-2D sequence assigning different values to different scales.
        '''
        if len(data.shape) < 2:
            data = data.clone().unsqueeze(0) # Note: do not call .detach(), it must stay attached to the computational graph

        # Save reference values for backward pass
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(       data, weights, bias)
        ctx.min_window_exponential = min_window_exponential
        ctx.max_window_exponential = max_window_exponential
        ctx.scale_multiplier       = scale_multiplier

        # Precompute values for a faster forward loop
        window  = scale_multiplier ** min_window_exponential
        num_win = data.shape[1] // (scale_multiplier ** min_window_exponential)

        # Define output variables
        # output       = torch.zeros(( data.shape[0] , (scale_multiplier ** min_window_exponential)*num_win*scale_multiplier -1 ), device=data.device, requires_grad=True) # TODO MAKE OUTPUT SMALLER, NEEDED THIS TO COMPARE LOSS
        output       = torch.zeros(( data.shape[0] , num_win*scale_multiplier -1 ), device=data.device, requires_grad=True) # TODO MAKE OUTPUT SMALLER, NEEDED THIS TO COMPARE LOSS
        output_index = data.shape[1] // window

        # Run the same operation for each scale, starting with the maximum window size
        for current_scale in range( min_window_exponential, max_window_exponential+1):
            # Retrieve appropriate weights and data
            kernel   = weights[window -1 : window * scale_multiplier -1 ]
            data_win = data.reshape(-1, window).transpose(0,1)

            # Calculate the output
            conv_out = kernel.matmul( data_win).unsqueeze(0)
            if bias is not None:
                conv_out += bias[ max_window_exponential - current_scale ]

            # Reshape and save output to the output array
            out = conv_out.reshape(-1, output_index)
            output[ : , output_index -1 : output_index*scale_multiplier -1 ] = out

            # Increment/decrement tracking variables appropriately
            window         = window * scale_multiplier
            output_index   = output_index // scale_multiplier

        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        '''backward method defining the backward pass operations. Relates the gradient values to each of the input parameters.

        Args:
            grad_output (Tensor): 1D Tensor holding the input data representing the gradient of this functions output.
        
        Returns:
            data (Tensor):        1D Tensor holding the gradient of data values after the most recent forward pass.
            weights (Tensor):     1D Tensor holding the gradient of weight values after the most recent forward pass. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):        1D Tensor holding the gradient of bias values after the most recent forward pass.
            None:                 Placeholder return value, represting the "gradient" of min_window_exponential
            None:                 Placeholder return value, represting the "gradient" of max_window_exponential
            None:                 Placeholder return value, represting the "gradient" of scale_multiplier
        '''
        # This function has only a single output, so it gets only one gradient
        if grad_output is None:
            return None, None, None, None, None, None

        # Extract the saved tensors for easier reference
        data, weights, bias    = ctx.saved_tensors
        max_window_exponential = ctx.max_window_exponential
        min_window_exponential = ctx.min_window_exponential
        scale_multiplier       = ctx.scale_multiplier

        # Precompute values for a faster forward loop
        window        = scale_multiplier ** min_window_exponential
        output_len    = (data.shape[1]+1) // window

        # Define output variables
        grad_data       = torch.zeros_like(data, device=data.device)
        grad_weights    = torch.zeros_like(weights, device=weights.device)
        grad_bias       = None if bias is None else torch.zeros_like(bias, device=bias.device)

        # Run the same operation for each scale, starting with the maximum window size
        for current_scale in range(min_window_exponential, max_window_exponential+1):
            # Extract the data relevant to the scale of interest.
            grad_out_scale = grad_output[ : , output_len-1 : output_len*scale_multiplier -1 ]
            weights_scale  = weights[window -1 : window * scale_multiplier -1 ].unsqueeze(0)
            data_scale     = data.reshape(-1, window) # Skip the last transpose

            ## Calculate grad_data
            # This execution keeps the batches separate, as required by pyTorch's Autograd function
            # grad_out_scale  = grad_out_scale.reshape((-1, 1))
            # grad_data_scale = grad_out_scale.matmul( weights_scale)
            # grad_data       = grad_data.add( grad_data_scale.reshape( -1, data.shape[1]))
            grad_data       = grad_data.add( grad_out_scale.reshape((-1, 1)).matmul( weights_scale).reshape( -1, data.shape[1]))

            # # This execution is fastest, but combines all the batches into one update.
            # # Is not compatible with autograd, as input/output dimensions do not match
            # weights_expanded = weights_scale.expand((grad_out_scale.shape[1], -1))
            # grad_data = grad_data.add( grad_out_scale.matmul( weights_expanded).view((-1,1)))

            ## Calculate grad_weights
            # grad_out_scale = grad_out_scale.reshape((1,-1))
            # grad_weights[window -1 : window * scale_multiplier -1 ] = grad_out_scale.matmul(data_scale)
            grad_weights[window -1 : window * scale_multiplier -1 ] = grad_out_scale.reshape((1,-1)).matmul(data_scale)

            ## Calculate grad_bias
            if bias is not None:
                grad_bias[ max_window_exponential - current_scale ] = grad_out_scale.sum()

            ## Increment/decrement tracking variables appropriately
            window         = window * scale_multiplier
            output_len     = output_len // scale_multiplier

        return grad_data, grad_weights, grad_bias, None, None, None



class TiScFunction_hidden(torch.autograd.Function):
    ''' Class defining static forward() and backward() methods for use by :py:class:`~toast.heat.timescaleL.TSC_hidden` , a  ``nn.Module``-based layer definition
    '''
    
    @staticmethod
    def forward(ctx, data, weights, bias, min_window_exponential, max_window_exponential, input_size_exponential, inclusive, scale_multiplier=2):
        '''forward method defining the forward pass operations.
        Bias must be included, but its value may be ``None``
        Important:
            Assumes input data is of dim=2 where dimensions represent (data_size, batch_size)
        Note:
            It is up to the user to verfify that the lengths of data samples (``data.shape[0]``) is an even multiple of the max_window size.
            Invalid dimension errors during windowing/reshaping commands will be raised if this is not true.
        Args:
            data (Tensor):                          1D Tensor holding the input data. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            weights (Tensor):                       1D Tensor holding the weights. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):                          1D Tensor holding the bias values.
            min_window_exponential (int):           Exponent value designating the minimum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            max_window_exponential (int):           Exponent value designating the maximum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            input_size_exponential (int):           Exponent value designating the exponential of the input data size. Should be defined consistent with the scale_multiplier variable.
            inclusive (bool):                       Whether scaled convolution operator should include time-relevant components from larger scales (longer windows)
            input_size_exponential (int, optional): Exponential needed over scale_multiplier to reach the size of the data. Important for internal index tracking, can be passed in to avoid in-line calculation.
            scale_multiplier (int, optional):       Exponential base designating how fast the window size increases.
        
        Returns:
            1D Tensor representing the output. Note these values should be interpreted in a pseudo-2D sequence assigning different values to different scales.
        '''
        if len(data.shape) < 2:
            data = data.clone().unsqueeze(1) # Note: do not call .detach(), it must stay attached to the computational graph

        # Save reference values for backward pass
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(       data, weights, bias)
        ctx.min_window_exponential = min_window_exponential
        ctx.max_window_exponential = max_window_exponential
        ctx.input_size_exponential = input_size_exponential
        ctx.inclusive              = inclusive
        ctx.scale_multiplier       = scale_multiplier

        # Define output variables
        output     = torch.zeros(( data.shape[0], scale_multiplier**(input_size_exponential-min_window_exponential) -1), device=data.device, requires_grad=True)
        output_idx = 2 ** (input_size_exponential-1 - min_window_exponential)

        # Run the same operation for each scale, starting with the maximum window size
        for current_scale in range( min_window_exponential, max_window_exponential+1):
            # Get the weights for this scale, and the associated activation values from the data passed in
            weights_scale       = TiScFunction_hidden._get_weights(  weights, current_scale, input_size_exponential, inclusive=inclusive)
            activations_stacked = TiScFunction_hidden._get_activations( data, current_scale, input_size_exponential, inclusive=inclusive)

            # Calculate the output
            activation_out = activations_stacked.matmul( weights_scale).unsqueeze(0)
            if bias is not None:
                activation_out += bias[ max_window_exponential - current_scale ]

            # Reshape and save the output to the output array
            output[ : , output_idx-1 : output_idx*scale_multiplier-1 ] = activation_out.reshape(data.shape[0], -1)

            # Increment/decrement tracking variables appropriately
            output_idx = output_idx // scale_multiplier

        return output
        

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        '''backward method defining the backward pass operations. Relates the gradient values to each of the input parameters.
        Args:
            grad_output (Tensor): 1D Tensor holding the input data representing the gradient of this functions output.
        
        Returns:
            data (Tensor):        1D Tensor holding the gradient of data values after the most recent forward pass.
            weights (Tensor):     1D Tensor holding the gradient of weight values after the most recent forward pass. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):        1D Tensor holding the gradient of bias values after the most recent forward pass.
            None:                 Placeholder return value, represting the "gradient" of min_window_exponential
            None:                 Placeholder return value, represting the "gradient" of max_window_exponential
            None:                 Placeholder return value, represting the "gradient" of inclusive
            None:                 Placeholder return value, represting the "gradient" of scale_multiplier
        '''
        # This function has only a single output, so it gets only one gradient
        if grad_output is None:
            return None, None, None, None, None, None, None, None

        # Extract the saved tensors for easier reference
        data, weights, bias = ctx.saved_tensors
        min_window_exponential = ctx.min_window_exponential
        max_window_exponential = ctx.max_window_exponential
        inclusive              = ctx.inclusive
        scale_multiplier       = ctx.scale_multiplier
        input_size_exponential  = ctx.input_size_exponential

        # Precompute values for a faster forward loop
        data_len            = scale_multiplier ** (input_size_exponential - (min_window_exponential +1))
        weights_len         = scale_multiplier ** (min_window_exponential +1)
        scales_remaining    = input_size_exponential - min_window_exponential -1 # needed if inclusive
        weights_offset_incl = weights.shape[0] - scales_remaining

        # Define output variables
        grad_data    = torch.zeros_like(data)
        grad_weights = torch.zeros_like(weights)
        grad_bias    = None if bias is None else torch.zeros_like(bias)

        # Run the same operation for each scale, starting with the minimum window size
        for current_scale in range(min_window_exponential, max_window_exponential+1):
            # Extract the data relevant to the scale of interest
            grad_out_scale = grad_output[ : , data_len-1 : data_len*scale_multiplier -1 ]
            weights_scale  = TiScFunction_hidden._get_weights(  weights, current_scale, input_size_exponential, inclusive=inclusive, scale_multiplier=scale_multiplier)
            data_scale     = TiScFunction_hidden._get_activations( data, current_scale, input_size_exponential, inclusive=inclusive, scale_multiplier=scale_multiplier)

            ## Calculate grad_data
            grad_data_scale     = torch.zeros_like(data)
            grad_out_scale      = grad_out_scale.reshape((-1, 1))
            grad_data_scale_raw = grad_out_scale.matmul( weights_scale.unsqueeze(0))
            if inclusive:
                # Handle the "exclusive" scale weights as usual
                raw_start_index  = 1                                                # Starting index in the normally sorted exclusive weights (shifted by scales_remaining)
                grad_start_index = data_len                                         # Starting index for the scale of interest
                end_start_index  = grad_data_scale_raw.shape[1] - scales_remaining  # Limit for the raw_start_index variable traching exclusive weights. Should iterate to total length minus scales_remaining
                while raw_start_index <= end_start_index:
                    for offset in range( raw_start_index):
                        # Note same as below, but grad_data_scale_raw is shifted by scales_remaining to account for the additional inclusive data positions used.
                        grad_data_scale[ : , grad_start_index-1+offset : grad_start_index*scale_multiplier-1+offset : raw_start_index ] = grad_data_scale_raw[ : , raw_start_index-1+offset+scales_remaining].reshape((-1, data_len))

                    raw_start_index  *= scale_multiplier
                    grad_start_index *= scale_multiplier

                # Condense the "inclusive" weigths, which are repeated accross multiple windows, to one update value using sum operator
                grad_data_scale_win = 1
                for incl_offset in range( scales_remaining):
                    # Retrieve the grad_data from this inclusive position, and reshape it so each row accounts for each inclusive window. (multiple adjacent outputs, incorporated by reshaping)
                    # Sum the gradients for each inclusive window (by summing along axis 1)
                    # Below is an equivalent implementation, which is more understandable but less efficient
                    # grad_data_scale_incl = grad_data_scale_raw[ : , incl_offset].reshape((-1,data_len//grad_data_scale_win))
                    # grad_data_scale_incl = grad_data_scale_incl.sum(1)
                    # for incl_win_offset in range(grad_data_scale_win):
                    #     # For each utilized offset, summ the values accross the outputs which utilized this inclusive weight. Each batch is separated by grad_data_scale_win
                    #     grad_data_scale[ : , grad_data_scale_win-1 + incl_win_offset ] = grad_data_scale_incl[ incl_win_offset::grad_data_scale_win]
                    grad_data_scale[ : , grad_data_scale_win-1 : (grad_data_scale_win*scale_multiplier) -1] = grad_data_scale_raw[ : , incl_offset].reshape((-1,data_len//grad_data_scale_win)).sum(1).reshape((-1,grad_data_scale_win))

                    grad_data_scale_win *= scale_multiplier

            else:
                # Handle the "exclusive" scale weights as usual
                raw_start_index  = 1
                grad_start_index = data_len
                end_start_index  = grad_data_scale_raw.shape[1]
                while raw_start_index <= end_start_index:
                    for offset in range( raw_start_index):
                        grad_data_scale[ : , grad_start_index-1+offset : grad_start_index*scale_multiplier-1+offset : raw_start_index ] = grad_data_scale_raw[ : , raw_start_index-1+offset].reshape((-1, data_len))

                    raw_start_index  *= scale_multiplier
                    grad_start_index *= scale_multiplier

            # Add the gradient for the data produced by this scale to the final grad_data array
            grad_data = grad_data.add( grad_data_scale)

            ## Calculate the grad_weights
            if inclusive:
                grad_weights_raw = grad_out_scale.transpose(0,1).matmul(data_scale)

                grad_weights[weights_len : weights_len * scale_multiplier -1 ]             = grad_weights_raw[ : , scales_remaining: ] # Exclusive Weights
                grad_weights[weights_offset_incl : weights_offset_incl + scales_remaining] = grad_weights_raw[ : , :scales_remaining ] # Inclusive Weights
                # Note: From above, inclusive weights are ordered |  ...exclusive weights...  | 1 |  2  |   3   |    4    ...
                #                                                                                   ^ within each scale, goes from large scale (low y position) to small scale

            else:
                grad_weights[weights_len : weights_len * scale_multiplier -1 ] = grad_out_scale.transpose(0,1).matmul(data_scale)

            ## Calculate the grad_bias
            if bias is not None:
                grad_bias[ max_window_exponential - current_scale ] = grad_out_scale.sum()

            ## Increment the tracker variables
            data_len             = data_len // scale_multiplier
            weights_len          = weights_len * scale_multiplier
            scales_remaining    -= 1
            weights_offset_incl -= scales_remaining
        
        return grad_data, grad_weights, grad_bias, None, None, None, None, None


    @staticmethod
    def _get_weights( weights, scale, input_size_expon, inclusive=False, scale_multiplier=2):
        ## We sort the scale-dependent weights in reverse order
        weights_start = scale_multiplier ** (scale + 1)

        if inclusive:
            scales_remaining = (input_size_expon - scale) -1
            output = torch.zeros((weights_start-1 + scales_remaining))
            # Weigths for the included scales are stored at the end of the structured weights array
            start_incl     = scale_multiplier ** (input_size_expon +1) - 1
            start_offset   = sum( range( 0, scales_remaining))
            # In the output weights, the included scale weights are inserted at the beginning
            # to match the ordering of how the activations are extracted
            output[:scales_remaining] = weights[ start_incl+start_offset : start_incl+start_offset+scales_remaining]
            
            output[scales_remaining:] = weights[ weights_start : weights_start*scale_multiplier -1]
        else:
            output                    = weights[ weights_start : weights_start*scale_multiplier -1]

        return output


    @staticmethod
    def _get_activations( array_in, scale, input_size_expon, inclusive=False, scale_multiplier=2):
        len_output  = scale_multiplier ** (scale + 1) - 1
        activ_index = scale_multiplier ** (input_size_expon - scale -1)

        if inclusive:
            scales_remaining = input_size_expon - scale -1
            output = torch.zeros((activ_index * array_in.shape[0], len_output+scales_remaining), device=array_in.device)
            output_index = scales_remaining - 1

            # Included activations start at the scale below the requested activation index
            activ_index_incl = int( activ_index / scale_multiplier)
            num_repeats      = scale_multiplier
            while activ_index_incl > 0:
                # Get activations for the next lowest scale
                included_activs         = array_in[ : , activ_index_incl -1 : (activ_index_incl * scale_multiplier) -1 ]
                # Repeat these activations for as many indexes as there are values in the smallest scale requested
                included_activs         = included_activs.reshape(-1, 1)
                output[ : , output_index] = included_activs.repeat_interleave( num_repeats)

                # Decrement indices to the lower scales
                activ_index_incl = activ_index_incl // scale_multiplier
                num_repeats      = num_repeats * scale_multiplier
                output_index     = output_index - 1
            
            output_index = scales_remaining
        else:
            output = torch.zeros((activ_index * array_in.shape[0], len_output), device=array_in.device)
            output_index = 0

        length_window = 1
        while( activ_index <=  array_in.shape[1]):
            activations_scale = array_in[ : , activ_index -1 : activ_index*scale_multiplier -1 ]
            activations_scale = activations_scale.reshape( -1, length_window)
            output[ : , output_index : output_index+length_window ] = activations_scale

            output_index  += length_window
            activ_index   = activ_index * scale_multiplier
            length_window = length_window * scale_multiplier
        
        return output


    @staticmethod
    def get_offset_activations( array_in, scale, offset, input_size_expon, inclusive=False, scale_multiplier=2):
        # Use if you only want part of a scale. 
        # The above function will grab all offsets of a particular scale and stack them. This only grabs one offset and returns it.
        len_output  = scale_multiplier ** (scale + 1) - 1
        activ_index = scale_multiplier ** (input_size_expon - scale -1) + offset

        if inclusive:
            output = torch.zeros(( array_in.shape[0], len_output+scale), device=array_in.device)
            output_index = scale - 1

            activ_index_incl = activ_index // scale_multiplier
            while activ_index_incl > 0:
                output[ : , output_index] = array_in[ : , activ_index_incl -1 ]
                activ_index_incl = activ_index_incl // scale_multiplier
                output_index     = output_index - 1
            
            output_index = scale
        else:
            output = torch.zeros(( array_in.shape[0], len_output), device=array_in.device)
            output_index = 0

        length_window = 1
        while( activ_index <= array_in.shape[1]):
            output[ : , output_index : output_index+length_window ] = array_in[ : , activ_index -1 : activ_index+length_window -1 ]
            output_index  += length_window
            activ_index   = activ_index * scale_multiplier
            length_window = length_window * scale_multiplier
        
        return output



class TiScTransposeFunction_input(torch.autograd.Function):
    ''' Class defining static forward() and backward() methods for use by :py:class:`~toast.heat.timescaleL.TSCTranspose_input`
    '''

    @staticmethod
    def forward(ctx, data, weights, bias, min_window_exponential, max_window_exponential, scale_multiplier=2):
        '''forward method defining the forward pass operations.
        Bias must be included, but its value may be ``None``
        Assumes that passed data is trimmed so that the output of min_window kernels is in the last half of values (no trailing zeros).
        Important:
            Assumes input data is of dim=2 where dimensions represent (data_size, batch_size)
        Note:
            It is up to the user to verfify that the lengths of data samples (``data.shape[0]``) is an even multiple of the max_window size.
            Invalid dimension errors during windowing/reshaping commands will be raised if this is not true.
        Args:
            data (Tensor):                    1D Tensor holding the input data.
            weights (Tensor):                 1D Tensor holding the weights. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):                    1D Tensor holding the bias values.
            min_window_exponential (int):     Exponent value designating the minimum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            max_window_exponential (int):     Exponent value designating the maximum window exponential to consider. Should be defined consistent with the scale_multiplier variable.
            scale_multiplier (int, optional): Exponential base designating how fast the window size increases.
        
        Returns:
            1D Tensor representing the output. Note these values should be interpreted in a pseudo-2D sequence assigning different values to different scales.
        '''
        if len(data.shape) < 2:
            data = data.clone().unsqueeze(1) # Note: do not call .detach(), it must stay attached to the computational graph

        # Save reference values for backward pass
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(       data, weights, bias)
        ctx.min_window_exponential = min_window_exponential
        ctx.max_window_exponential = max_window_exponential
        ctx.scale_multiplier       = scale_multiplier

        # Precompute values for a faster forward loop
        window     = scale_multiplier ** min_window_exponential
        num_win    = data.shape[1] - ((data.shape[1]+1) // scale_multiplier) + 1
        output_len = window * num_win
        data_index = (data.shape[1]+1) // scale_multiplier

        # Define output variables
        output = torch.zeros((data.shape[0], output_len), device=data.device, requires_grad=True)

        # Run the same operation for each scale, starting with the maximum window size
        for current_scale in range( min_window_exponential, max_window_exponential+1):
            # Retrieve appropriate weights and data
            kernel   = weights[window -1 : window * scale_multiplier -1 ].unsqueeze(0)
            data_scale = data[ : , data_index -1 : data_index*scale_multiplier -1].reshape(-1,1)

            # Calculate the output
            conv_out = data_scale.matmul(kernel)
            if bias is not None:
                conv_out += bias[ max_window_exponential - current_scale ]

            # Reshape and save output to the output array
            out = conv_out.reshape(-1,output_len)
            output += out

            # Increment/decrement tracking variables appropriately
            window     = window * scale_multiplier
            data_index = data_index // scale_multiplier

        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        '''backward method defining the backward pass operations. Relates the gradient values to each of the input parameters.
        Args:
            grad_output (Tensor): 1D Tensor holding the input data representing the gradient of this functions output.
        
        Returns:
            data (Tensor):        1D Tensor holding the gradient of data values after the most recent forward pass.
            weights (Tensor):     1D Tensor holding the gradient of weight values after the most recent forward pass. Note these values are interpreted in a pseudo-2D sequence assigning different values to different scales.
            bias (Tensor):        1D Tensor holding the gradient of bias values after the most recent forward pass.
            None:                 Placeholder return value, represting the "gradient" of min_window_exponential
            None:                 Placeholder return value, represting the "gradient" of max_window_exponential
            None:                 Placeholder return value, represting the "gradient" of scale_multiplier
        '''
        # This function has only a single output, so it gets only one gradient
        if grad_output is None:
            return None, None, None, None, None, None

        # Extract the saved tensors for easier reference
        data, weights, bias    = ctx.saved_tensors
        max_window_exponential = ctx.max_window_exponential
        min_window_exponential = ctx.min_window_exponential
        scale_multiplier       = ctx.scale_multiplier

        # Precompute values for a faster forward loop
        window        = scale_multiplier ** min_window_exponential
        data_index    = (data.shape[1]+1) // scale_multiplier

        # Define output variables
        grad_data       = torch.zeros_like(data, device=data.device)
        grad_weights    = torch.zeros_like(weights, device=weights.device)
        grad_bias       = None if bias is None else torch.zeros_like(bias, device=bias.device)

        # Run the same operation for each scale, starting with the maximum window size
        for current_scale in range(min_window_exponential, max_window_exponential+1):
            # Extract the data relevant to the scale of interest.
            grad_out_scale = grad_output.reshape(-1, window)
            weights_scale  = weights[window -1 : window * scale_multiplier -1 ].unsqueeze(0)
            data_scale     = data[ : , data_index -1 : data_index * scale_multiplier -1].reshape(1,-1).squeeze()

            ## Calculate grad_data
            grad_data[ : , data_index -1 : data_index * scale_multiplier -1] = weights_scale.matmul(grad_out_scale.transpose(0,1)).reshape(-1,data_index)

            ## Calculate grad_weights
            grad_weights[ window -1 : window * scale_multiplier -1] = data_scale.matmul(grad_out_scale)

            ## Calculate grad_bias
            if bias is not None:
                grad_bias[ max_window_exponential - current_scale ] = grad_out_scale.sum()

            ## Increment/decrement tracking variables appropriately
            window         = window * scale_multiplier
            data_index     = data_index // scale_multiplier

        return grad_data, grad_weights, grad_bias, None, None, None



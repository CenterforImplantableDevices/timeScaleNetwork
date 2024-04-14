'''
    A Class that contains all the functionality and tracking necessary to train and debug Deep Neural Networks.
'''
from timeScaleNetwork import timescaleF as tsf
from timeScaleNetwork import noise
import torch.nn as nn
import torch
import math



class TSC_input(nn.Module):
    '''Represents a time scale convolution input layer.

    Typically only one of these layers is present at the input.  It is designed to transofrm a 1D input into a 2D Scaled output.
    '''

    def __init__(self, min_window, max_window, scale_multiplier=2, bias=True, initialization=None):
        '''Initialization to create a Time Scale Convolutional Layer

        Args:
            min_window (int):                   The minimum window size to consider. Value represents the min window size in number of samples.
            max_window (int):                   The maximum window size to consider. Value represents the max window size in number of samples.
            scale_multiplier (int, optional):   The exponential base designating how fast the window size increases.
            bias (bool, optional):              Whether to add a bias node to each scale.
            initialization (int/str, optional): How to initializae tisc parameters. Can be a beta value for noise, or a string requesting a noise color. See :py:func:`toast.butter.noise.colors.get_noise` or :py:func:`toast.butter.noise.colors.get_color` for more info.
        
        Returns:
            An instance of the TSC_input class representing a Time Scale Convolution Layer.
        '''
        super(TSC_input, self).__init__()

        self.min_window       = min_window
        self.max_window       = max_window
        self.scale_multiplier = scale_multiplier
        if not (self.scale_multiplier == 2):
            print('ERROR: Only scale_multiplier=2 is currently supported. Setting scale_multiplier to 2.')
            self.scale_multiplier = 2

        # Determine the minimum and miximum exponential relting the the minimum/maximum window sizes.
        # Window sizes should be an even multiple of scale_multiplier, if not the windows will be reduced/expanded.
        if not math.log( self.min_window, self.scale_multiplier).is_integer():
            print('WARNING: min_window (', self.min_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Reducing min_window to compensate')
            self.min_window = self.scale_multiplier ** int( math.log( self.min_window, self.scale_multiplier))
            print('WARNING: New value of min_window: ', self.min_window)
        if not math.log( self.max_window, self.scale_multiplier).is_integer():
            print('WARNING: max_window (', self.max_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Expanding max_window to compensate')
            self.max_window = self.scale_multiplier ** int( math.log( self.max_window, self.scale_multiplier)+1)
            print('WARNING: New value of max_window: ', self.max_window)
        self.min_window_exponential = int( math.log( self.min_window, self.scale_multiplier))
        self.max_window_exponential = int( math.log( self.max_window, self.scale_multiplier))
        
        weights_length = self.max_window * self.scale_multiplier - 1
        self.weights = nn.Parameter( torch.Tensor( weights_length).requires_grad_())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.max_window_exponential).requires_grad_())
        else:
            self.register_parameter('bias', None)
        self.param_initialization = initialization
        self.reset_parameters()
        self.show_warning_data = False

    # TODO Not sure what this does, but it's in all the examples
    def reset_parameters(self) -> None:
        '''Resets the parameters to a default distribution that was passed in __init__.
        
        Weights are normally distributed, unless a noise pattern/color is selected during intialization.
        Bias is uniformly distributed.
        
        Args:
            None.
        
        Returns:
            No return value. Weight/Bias values will be reset.
        '''
        stdev = 1 / math.sqrt( self.weights.shape[-1])
        if self.param_initialization is None:
            nn.init.uniform_(self.weights, -stdev,stdev)
        else:
            with torch.no_grad():
                if isinstance( self.param_initialization, int) or isinstance( self.param_initialization, float):
                    self.weights.data = torch.Tensor( stdev * noise.get_noise( self.weights.shape, beta=self.param_initialization))
                elif isinstance( self.param_initialization, str):
                    self.weights.data = torch.Tensor( stdev * noise.get_color( self.weights.shape, self.param_initialization))
                else:
                    print('WARNGING: Invalid initialization argument. Initializing to uniform distribution.')
                    nn.init.uniform_(self.weights, -stdev,stdev)

        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            # bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -stdev, stdev)
    

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.
        
        Args:
            data (tensor): Data to be processed by the network.
        
        Returns:
            The output when passing the input data through the network
        '''
        return tsf.TiScFunction_input.apply( self._check_dataShape(data), self.weights, self.bias, self.min_window_exponential, self.max_window_exponential, self.scale_multiplier)

    def convert_to_decision( self, output):
        '''This method overwrites the forward method of the Network class, in case special behavior is desired
        Convertrs a network output to a definite decision based on thresholds highest likelihood values.
        
        Args:
            output: Value output by network
        
        Returns:
            Float Tensor representing the final decision of the network.
        '''
        if self.one_hot_output:
            return (torch.argmax( output, dim=-1)).float()
        else:
            return (output > self.decision_threshold).float()

    def extra_repr(self):
        '''Set the extra information about this module. 
        '''
        return 'min_window={}, max_window={}, mult={}, bias={}'.format(
            self.min_window, self.max_window, self.scale_multiplier, self.bias is not None
        )

    def freeze_shape(self, freeze):
        '''Note this function does NOT freeze the bias, only the weights'''
        if freeze:
            self.weights.requires_grad = False
        else:
            self.weights.requires_grad = True

    def get_gradCAM( self, activations, activations_grad, input_dimensions=None):
        '''Calculate gradCAM activation mapping for this layer, rescaled and stretched to match the input dimensions.

        Currently, the activations and activation gradients must be extracted from the larger network object outside of this function. An example imlementation is below::

            activations      = network.layer_activations[layer]
            activations_grad = network.layer_activationsGrad[layer]
            gradCAM_out, gradCAM_expanded, gradCAM_alpha = layer.get_gradCAM(activations.to(network.device), activations_grad.to(network.device), input_dimensions=sample_shape)
        
        Args:
            activations (Tensor):                  Activation values of layer after forward pass is performed. To track these, use :py:func:`toast.heat.network.add_hook_activation`
            activations_grad (Tensor):             Gradient of the activation values after a backwards pass is performed. To track these, use :py:func:`toast.heat.network.add_hook_activationGrad`
            input_dimensions (iterable, optional): Shape of the input dimensions to match. If no input dimensions are passed, the largest window of this layer is assumed to be the length of the input.
        
        Returns:
            *gradCAM_output* - gradCAM results which match shape of input sample; *gradCAM_expanded* - gradCAM results which are expanded to allow visualizaiton of activations of each independent scale; *scale_alpha* - Alpha values for each scale, signifying importance.
        '''
        # Precompute values for a faster forward loop
        scale_multiplier  = self.scale_multiplier
        min_win_expon     = self.min_window_exponential
        max_win_expon     = self.max_window_exponential
        current_scale     = max_win_expon

        if input_dimensions is None:
            input_dimensions = [scale_multiplier ** max_win_expon]
        index_activation  = input_dimensions[-1] // (scale_multiplier ** max_win_expon)

        scale_alpha      = torch.zeros(activations.shape[0], max_win_expon-min_win_expon+1) # This is it's own matrix in case it is desired for analysis
        gradCAM_expanded = torch.zeros(activations.shape[0], max_win_expon-min_win_expon+1, input_dimensions[-1] // (scale_multiplier ** min_win_expon))
        while current_scale >= min_win_expon:
            index_start = index_activation -1
            index_end   = index_activation * scale_multiplier -1
            # alpha = activations_weighted[: , index_start:index_end].sum(axis=-1)
            alpha = activations_grad[: , index_start:index_end].sum(axis=-1)
            scale_alpha[:, current_scale-min_win_expon] = alpha
            
            gradCAM_expanded[:, current_scale-min_win_expon, :] = activations[: , index_start:index_end].transpose(0,1).multiply(alpha).transpose(0,1).repeat_interleave(scale_multiplier**(current_scale-min_win_expon), dim=-1)

            # Increment/decrement tracking variables appropriately
            index_activation = index_activation * scale_multiplier
            current_scale   -= 1

        # Dimension order should be [samples, scales, time]
        repeat_matchInput = input_dimensions[-1] // gradCAM_expanded.shape[-1]
        fcn_relu          = nn.ReLU()
        gradCAM_out       = fcn_relu(gradCAM_expanded.sum(dim=-2)).repeat_interleave(repeat_matchInput, axis=-1)
        gradCAM_expanded  = fcn_relu(gradCAM_expanded            ).repeat_interleave(repeat_matchInput, axis=-1)
        
        return gradCAM_out, gradCAM_expanded, scale_alpha


    def get_gradCAM_single( self, activations, activations_grad, input_dimensions=None):
        '''Calculate gradCAM activation mapping for this layer, rescaled and stretched to match the input dimensions. It is recommended to use :py:func:`get_gradCAM` instead of this function. This is maintained for backwards compatibility.

        Currently, the activations and activation gradients must be extracted from the larger network object outside of this function. An example imlementation is below::

            activations      = network.layer_activations[layer]
            activations_grad = network.layer_activationsGrad[layer]
            gradCAM_out, gradCAM_expanded, gradCAM_alpha = layer.get_gradCAM(activations.to(network.device), activations_grad.to(network.device), input_dimensions=sample_shape)
        
        Args:
            activations (Tensor):                  Activation values of layer after forward pass is performed. To track these, use :py:func:`toast.heat.network.add_hook_activation`
            activations_grad (Tensor):             Gradient of the activation values after a backwards pass is performed. To track these, use :py:func:`toast.heat.network.add_hook_activationGrad`
            input_dimensions (iterable, optional): Shape of the input dimensions to match. If no input dimensions are passed, the largest window of this layer is assumed to be the length of the input.
        
        Returns:
            *gradCAM_output* - gradCAM results which match shape of input sample; *gradCAM_expanded* - gradCAM results which are expanded to allow visualizaiton of activations of each independent scale; *scale_alpha* - Alpha values for each scale, signifying importance.
        '''
        # Precompute values for a faster forward loop
        scale_multiplier  = self.scale_multiplier
        min_win_expon     = self.min_window_exponential
        max_win_expon     = self.max_window_exponential
        current_scale     = max_win_expon

        if input_dimensions is None:
            input_dimensions = [scale_multiplier ** max_win_expon]
        index_activation  = input_dimensions[-1] // (scale_multiplier ** max_win_expon)

        scale_alpha = torch.zeros(max_win_expon-min_win_expon+1) # This is it's own matrix in case it is desired for analysis
        gradCAM_expanded = torch.zeros(max_win_expon-min_win_expon+1, input_dimensions[-1] // (scale_multiplier ** min_win_expon))
        while current_scale >= min_win_expon:
            index_start = index_activation -1
            index_end   = index_activation * scale_multiplier -1
            # alpha = activations_weighted[index_start : index_end].sum()
            alpha = activations_grad[index_start : index_end].sum()

            scale_alpha[current_scale-min_win_expon]        = alpha
            gradCAM_expanded[current_scale-min_win_expon,:] = activations[index_start:index_end].multiply(alpha).repeat_interleave(scale_multiplier**(current_scale-min_win_expon))
            
            # Increment/decrement tracking variables appropriately
            index_activation = index_activation * scale_multiplier
            current_scale   -= 1

        repeat_matchInput = input_dimensions[-1] // gradCAM_expanded.shape[-1]
        fcn_relu          = nn.ReLU()
        gradCAM_out       = fcn_relu(gradCAM_expanded.sum(axis=-2)).repeat_interleave(repeat_matchInput, axis=-1)
        gradCAM_expanded  = fcn_relu(gradCAM_expanded             ).repeat_interleave(repeat_matchInput, axis=-1)
        
        return gradCAM_out, gradCAM_expanded, scale_alpha

    def _check_dataShape( self, data):
        if len( data.shape) == 1:
            if data.shape[0] % self.max_window != 0:
                if not self.show_warning_data:
                    print('WARNING: input data is not a multiple of max_window. Truncating data to a valid length.')
                    self.show_warning_data = True
                data = data[ :-1*(data.shape[0] % self.max_window)]
        else:
            if data.shape[1] % self.max_window != 0:
                if not self.show_warning_data:
                    print('WARNING: input data is not a multiple of max_window. Truncating data to a valid length.')
                    self.show_warning_data = True
                data = data[ : , :-1*(data.shape[1] % self.max_window) ]
        
        return data



class TSCTranspose_input(nn.Module):
    '''Represents a transposed time scale convolution input layer.
    Typically used to build a mirrored Decoder structure in an time scale encoder/decoder network.

    returns an encoded vector back to input dimensions.
    '''

    def __init__(self, min_window, max_window, scale_multiplier=2, bias=True, initialization=None):
        '''Initialization to create a Transposed Time Scale Convolutional Layer.

        Args:
            min_window (int):                   The minimum window size to consider. Value represents the min window size in number of samples.
            max_window (int):                   The maximum window size to consider. Value represents the max window size in number of samples.
            scale_multiplier (int, optional):   The exponential base designating how fast the window size increases.
            bias (bool, optional):              Whether to add a bias node to each scale.
            initialization (int/str, optional): How to initializae tisc parameters. Can be a beta value for noise, or a string requesting a noise color. See :py:func:`toast.butter.noise.colors.get_noise` or :py:func:`toast.butter.noise.colors.get_color` for more info.
        
        Returns:
            An instance of the TSC_input class representing a Time Scale Convolution Layer.
        '''
        super(TSCTranspose_input, self).__init__()

        self.min_window       = min_window
        self.max_window       = max_window
        self.scale_multiplier = scale_multiplier
        if not (self.scale_multiplier == 2):
            print('ERROR: Only scale_multiplier=2 is currently supported. Setting scale_multiplier to 2.')
            self.scale_multiplier = 2

        # Determine the minimum and miximum exponential relting the the minimum/maximum window sizes.
        # Window sizes should be an even multiple of scale_multiplier, if not the windows will be reduced/expanded.
        if not math.log( self.min_window, self.scale_multiplier).is_integer():
            print('WARNING: min_window (', self.min_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Reducing min_window to compensate')
            self.min_window = self.scale_multiplier ** int( math.log( self.min_window, self.scale_multiplier))
            print('WARNING: New value of min_window: ', self.min_window)
        if not math.log( self.max_window, self.scale_multiplier).is_integer():
            print('WARNING: max_window (', self.max_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Expanding max_window to compensate')
            self.max_window = self.scale_multiplier ** int( math.log( self.max_window, self.scale_multiplier)+1)
            print('WARNING: New value of max_window: ', self.max_window)
        self.min_window_exponential = int( math.log( self.min_window, self.scale_multiplier))
        self.max_window_exponential = int( math.log( self.max_window, self.scale_multiplier))
        
        weights_length = self.max_window * self.scale_multiplier - 1
        self.weights = nn.Parameter( torch.Tensor( weights_length).requires_grad_())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.max_window_exponential).requires_grad_())
        else:
            self.register_parameter('bias', None)
        self.param_initialization = initialization
        self.reset_parameters() 
        self.show_warning_data = False

    # TODO Not sure what this does, but it's in all the examples
    def reset_parameters(self) -> None:
        '''Resets the parameters to a default distribution that was passed in __init__.
        
        Weights are normally distributed, unless a noise pattern/color is selected during intialization.
        Bias is uniformly distributed.
        
        Args:
            None.
        
        Returns:
            No return value. Weight/Bias values will be reset.
        '''
        stdev = 1 / math.sqrt( self.weights.shape[-1])
        if self.param_initialization is None:
            nn.init.uniform_(self.weights, -stdev,stdev)
        else:
            with torch.no_grad():
                if isinstance( self.param_initialization, int) or isinstance( self.param_initialization, float):
                    self.weights.data = torch.Tensor( stdev * noise.get_noise( self.weights.shape, beta=self.param_initialization))
                elif isinstance( self.param_initialization, str):
                    self.weights.data = torch.Tensor( stdev * noise.get_color( self.weights.shape, self.param_initialization))
                else:
                    print('WARNGING: Invalid initialization argument. Initializing to uniform distribution.')
                    nn.init.uniform_(self.weights, -stdev,stdev)

        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            # bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -stdev, stdev)
    

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.
        
        Args:
            data (tensor): Data to be processed by the network.
        
        Returns:
            The output when passing the input data through the network
        '''
        return tsf.TiScTransposeFunction_input.apply( data, self.weights, self.bias, self.min_window_exponential, self.max_window_exponential, self.scale_multiplier)

    def convert_to_decision( self, output):
        '''This method overwrites the forward method of the Network class, in case special behavior is desired
        Convertrs a network output to a definite decision based on thresholds highest likelihood values.
        
        Args:
            output: Value output by network
        
        Returns:
            Float Tensor representing the final decision of the network.
        '''
        if self.one_hot_output:
            return (torch.argmax( output, dim=-1)).float()
        else:
            return (output > self.decision_threshold).float()

    def extra_repr(self):
        '''Set the extra information about this module. 
        '''
        return 'min_window={}, max_window={}, mult={}, bias={}'.format(
            self.min_window, self.max_window, self.scale_multiplier, self.bias is not None
        )



class TSC_hidden(nn.Module):
    '''Represents a time scale convolution hidden layer
        
        Typically these layers are stacked as hidden layers in a nerual network. It is designed to transform a 2D scaled output into another 2D scaled output.
    '''

    def __init__(self, min_window, max_window, input_size, inclusive, scale_multiplier=2, bias=True, initialization=None):
        '''Initialization to create a Time Scale Convolutional Hidden Layer.
        Args:
            min_window (int):                   TODO Should this be the scale/exponential???? The minimum window size to consider. Value represents the min window size in number of samples.
            max_window (int):                   TODO Should this be the scale/exponential???? The maximum window size to consider. Value represents the max window size in number of samples.
            inclusive (bool):                   Whether this layer should include larger scales at a given time point when forward propogating features.
            scale_multiplier (int, optional):   The exponential base designating how fast the window size increases.
            bias (bool, optional):              Whether to add a bias node to each scale.
            initialization (int/str, optional): How to initializae tisc parameters. Can be a beta value for noise, or a string requesting a noise color.
        Returns:
            An instance of the TSC_hidden class representing a Time Scale Convolution Hidden Layer.
        '''
        super(TSC_hidden, self).__init__()

        self.min_window            = min_window
        self.max_window            = max_window
        self.input_size            = input_size
        self.scale_multiplier      = scale_multiplier
        self.inclusive             = inclusive
        if not (self.scale_multiplier == 2):
            print('ERROR: Only scale_multiplier=2 is currently supported. Setting scale_multiplier to 2.')
            self.scale_multiplier = 2

        # Determine the minimum and miximum exponential relting the the minimum/maximum window sizes.
        # Window sizes should be an even multiple of scale_multiplier, if not the windows will be reduced/expanded.
        if not math.log( self.min_window, self.scale_multiplier).is_integer():
            print('WARNING: min_window (', self.min_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Reducing min_window to compensate')
            self.min_window = self.scale_multiplier ** int( math.log( self.min_window, self.scale_multiplier))
            print('WARNING: New value of min_window: ', self.min_window)
        if not math.log( self.max_window, self.scale_multiplier).is_integer():
            print('WARNING: max_window (', self.max_window, ') was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Expanding max_window to compensate')
            self.max_window = self.scale_multiplier ** int( math.log( self.max_window, self.scale_multiplier)+1)
            print('WARNING: New value of max_window: ', self.max_window)
        if not math.log( self.input_size +1, self.scale_multiplier).is_integer():
            print('WARNING: input_size (', self.input_size, ' +1) was not an even multiple of scale_multiplier (', self.scale_multiplier, '). Reducing input_size to compensate')
            self.input_size = self.scale_multiplier ** int( math.log( self.input_size +1, self.scale_multiplier)) -1
            print('WARNING: New value of input_size: ', self.input_size)
        self.min_window_exponential = int( math.log( self.min_window,    self.scale_multiplier))
        self.max_window_exponential = int( math.log( self.max_window,    self.scale_multiplier))
        self.input_size_exponential = int( math.log( self.input_size +1, self.scale_multiplier))
        # num_windows                 = self.max_window_exponential - self.min_window_exponential
        
        weights_length = (self.max_window*self.scale_multiplier) * self.scale_multiplier - 1
        if inclusive:
            num_scales = self.max_window_exponential - self.min_window_exponential +1
            weights_length += sum( range( num_scales)) # Number of scales
        self.weight = nn.Parameter( torch.Tensor( weights_length).requires_grad_())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.max_window_exponential+1).requires_grad_())
            # self.bias = nn.Parameter(torch.Tensor(num_windows))
        else:
            self.register_parameter('bias', None)
        self.param_initialization = initialization
        self.reset_parameters() 
        self.show_warning_data = False

        self.gradCAM_latest           = None
        self.gradCAM_expanded_latest  = None
        self.gradCAM_alpha_latest     = None

    # TODO Not sure what this does, but it's in all the examples
    def reset_parameters(self) -> None:
        '''Resets the parameters to a default distribution that was passed in __init__.
        
        Weights are normally distributed, unless a noise pattern/color is selected during intialization.
        Bias is uniformly distributed.
        
        Args:
            None.
        
        Returns:
            No return value. Weight/Bias values will be reset.
        '''
        stdev = 1 / math.sqrt( self.weight.shape[-1])
        if self.param_initialization is None:
            nn.init.uniform_(self.weight, -stdev,stdev)
        else:
            with torch.no_grad():
                if isinstance( self.param_initialization, int) or isinstance( self.param_initialization, float):
                    self.weight.data = torch.Tensor( stdev * noise.get_noise( self.weight.shape, beta=self.param_initialization))
                elif isinstance( self.param_initialization, str):
                    self.weight.data = torch.Tensor( stdev * noise.get_color( self.weight.shape, self.param_initialization))
                else:
                    print('WARNGING: Invalid initialization argument. Initializing to uniform distribution.')
                    nn.init.uniform_(self.weight, -stdev,stdev)

        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -stdev, stdev)
    

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.
        Args:
            data (tensor): Data to be processed by the network.
        Returns:
            The output when passing the input data through the network
        '''
        return tsf.TiScFunction_hidden.apply( data, self.weight, self.bias, self.min_window_exponential, self.max_window_exponential, self.input_size_exponential, self.inclusive, self.scale_multiplier)


    def convert_to_decision( self, output):
        '''This method overwrites the forward method of the Network class, in case special behavior is desired
        Convertrs a network output to a definite decision based on thresholds highest likelihood values.
        Args:
            output: Value output by network
        Returns:
            Float Tensor representing the final decision of the network.
        '''
        if self.one_hot_output:
            return (torch.argmax( output, dim=-1)).float()
        else:
            return (output > self.decision_threshold).float()


    def extra_repr(self):
        '''Set the extra information about this module. 
        '''
        return 'min_window={}, max_window={}, input_size={}, input_size_exp={}, inclusive={}, mult={}, bias={}'.format(
            self.min_window, self.max_window, self.input_size, self.input_size_exponential, self.inclusive, self.scale_multiplier, self.bias is not None
        )



if __name__ == "__main__":

    testNet = TSC_input( 4+1, 16384-1, scale_multiplier=2, bias=True)
    
    print('Network:')
    print(testNet)

    print('Fin.')

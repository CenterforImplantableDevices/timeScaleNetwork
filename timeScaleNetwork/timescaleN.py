'''
    A Class that contains all the functionality and tracking necessary to train and debug Deep Neural Networks.
'''
from timeScaleNetwork import network as toast_network
from timeScaleNetwork import timescaleL as tsl
from timeScaleNetwork import perceptronN
import torch.nn as nn
import torch



class TiscMlpN(toast_network.Network):
    def __init__(self, layer_size, num_tisc_channels=1, length_input=None, dropout=[], dropout_perc=0.2, nonlinearity_tisc=nn.ReLU, nonlinearity_tiscHidden=None, nonlinearity=nn.ReLU, nonlinearity_final=nn.Sigmoid, tisc_initialization=None, tisc_scale_multiplier=2, path_save_progress='./', description='', comment='none'):
        '''Combines a Time Scale Input :py:func:`~toast.heat.timescaleL.TSC_input` layer encoder with a mutlilayer perceptron `~toast.heat.perceptronN.PerceptronN` decision/classifier component.

        If length_input is set to `None`, then the largest window size is assumed to be the length of the input.

        Args:
            layer_size (1D iterable):           Sequential list of the size of each layer. 
                                                The first index is a 2 element list representing [min_window, max_window]. The remaining elements are used to construct a mlp using the indicated layer sizes
            num_tisc_channels (int, optional):  How many repetitions or independent channels of Time Scale layers to apply to the data.
            length_input (int, optional):       Size of the data that will be input into the network. Necessary if the largest TiSc window size does not cover the entire data length, to aid in dimension calculation.
            dropout (1D iterable, optional):    Iterable of Boolean Values: Sequentially decide if the correlating layer in layer_size iterable should be followed by a dropout layer. Passed to MLP initialization only. If value(s) don't exist they will be set to False.
            dropout_perc (float, optional):     Probability of an element to be zeroed.
            nonlinearity_tisc (torch activation, optional):  pytorch.nn activation fuction to use within the TiSc operation. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            nonlinearity (torch activation, optional):       pytorch.nn activation fuction to use. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            nonlinearity_final (torch activation, optional): pytorch.nn activation fuction to use after the final layer. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            tisc_initialization (int or str, optional):      Initialization argument defining TiSc parameter initialization settings. See :py:func:`~toast.heat.timescaleL.TSC_input.__init__` for more information.
            tisc_scale_multiplier (int, optional):           Exponential base designating how fast the window size increases.
            path_save_progress (str, optional): Path to the parent directory where progress and tracking data should be saved.
            description (str, optional):        A description describing the contents/structure of the network for easy reference.
            comment (str, optional):            A comment about what this network is being used to do/test/investigate for later identification of tests run.

        Returns:
            An instance of the PerceptronNN class representing a deep neural network.
        '''
        # nonlinearity = nn.Tanh

        self.decision_threshold  = 0.5
        self.one_hot_output = True if (layer_size[-1] > 1) else False

        # Define the expected layer types, used for organizing potential debugging output.
        layer_type_dict = {'TiScLayer':    [tsl.TSC_input],
                           'TiScHidLayer': [tsl.TSC_hidden],
                           'NormLayer':    [nn.BatchNorm1d],
                           'ConnLayer':    [nn.Linear], 
                           'NonlinLayer':  [nn.Tanhshrink, nonlinearity, nonlinearity_final]} # Dropout layer type not identified, will be categorized as "Other"
        # Call Network __init__() to initialize standard network variables and functions
        super(TiscMlpN, self).__init__( layer_type_dict=layer_type_dict, path_save_progress=path_save_progress, description=description, description_short='tiscmlp', comment=comment)

        # Define the linear layers
        tisc_channels  = []
        for _ in range(num_tisc_channels):
            layers_temp = []
            # TiSc Layers
            tisc_temp = tsl.TSC_input( layer_size[0][0], layer_size[0][1], scale_multiplier=tisc_scale_multiplier, bias=True, initialization=tisc_initialization)
            layers_temp.append(tisc_temp)
            # Nonlinearity
            if nonlinearity_tisc is not None:
                layers_temp.append(nonlinearity_tisc())
            # Dropout Layers # TODO ADD IF STATEMENT
            layers_temp.append(nn.Dropout(p=dropout_perc))
            # Finalize layers as sequential object
            tisc_channels.append( layers_temp)
        
        # Define MLP part of network
        if length_input is None:
            tisc_outputLen_perCh = (layer_size[index_layer-1][1] * tisc_scale_multiplier -1) # * num_tisc_channels
        else:
            tisc_outputLen_perCh = ((length_input // tisc_temp.min_window) * tisc_scale_multiplier -1) # * num_tisc_channels

        index_layer   = 1
        while isinstance( layer_size[index_layer], list):
            for num_channel in range(num_tisc_channels):
                tisc_temp = tsl.TSC_hidden( layer_size[index_layer][0], layer_size[index_layer][1], tisc_outputLen_perCh, layer_size[index_layer][2], scale_multiplier=tisc_scale_multiplier, bias=True, initialization=tisc_initialization)
                tisc_channels[num_channel].extend([tisc_temp])
                # Nonlinearity
                if nonlinearity_tiscHidden is not None:
                    tisc_channels[num_channel].extend([nonlinearity_tiscHidden()])
                # Dropout Layers
                tisc_channels[num_channel].extend([nn.Dropout(p=dropout_perc)])
            tisc_outputLen_perCh = ((tisc_outputLen_perCh +1) // tisc_temp.min_window) -1

            index_layer += 1
        
        tisc_channels = [nn.Sequential( *channel) for channel in tisc_channels]
        tisc_layers   = nn.Sequential( *tisc_channels)

        mlp_layerSize = [tisc_outputLen_perCh * num_tisc_channels]
        mlp_layerSize.extend(layer_size[index_layer:])
        mlp_layers    = perceptronN.define_sequential_linear_layers( mlp_layerSize, dropout=dropout, dropout_perc=dropout_perc, nonlinearity=nonlinearity, nonlinearity_final=nonlinearity_final)
        
        # Combine TiSc and MLP components in one sequential object
        self.layers   = nn.Sequential( tisc_layers, mlp_layers)
        self.new_layers_added()
        

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.

        Args:
            data (tensor): Data to be processed by the network.

        Returns:
            The output when passing the input data through the network
        '''

        if len(self.layers[0]) == 1:
            tisc_output = self.layers[0][0](data)
        else:
            tisc_output_channels = []
            for channel in self.layers[0]:
                tisc_output_channels.append( channel( data))
            tisc_output = torch.cat( tisc_output_channels, dim=-1)
            
        return self.layers[1]( tisc_output.reshape( tisc_output.shape[0], -1))


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



class TiscEncDecN(toast_network.Network):
    def __init__(self, layer_size, num_tisc_channels=1, length_input=None, dropout=[], dropout_perc=0.2, nonlinearity=nn.ReLU, tisc_initialization=None, tisc_scale_multiplier=2, path_save_progress='./', description='', comment='none'):
        '''Combines a Time Scale Input :py:func:`~toast.heat.timescaleL.TSC_input` encoder layer with a Transposed Time Scale Input `~toast.heat.timescaleL.TSCTranspose_input` decoder layer.

        If length_input is set to `None`, then the largest window size is assumed to be the length of the input.

        Args:
            layer_size (1D iterable):           Sequential list of the size of each layer. 
                                                The first index is a 2 element list representing [min_window, max_window]. The remaining elements are used to construct a mlp using the indicated layer sizes
            num_tisc_channels (int, optional):  How many repetitions or independent channels of Time Scale layers to apply to the data.
            length_input (int, optional):       Size of the data that will be input into the network. Necessary if the largest TiSc window size does not cover the entire data length, to aid in dimension calculation.
            dropout (1D iterable, optional):    Iterable of Boolean Values: Sequentially decide if the correlating layer in layer_size iterable should be followed by a dropout layer. Passed to MLP initialization only. If value(s) don't exist they will be set to False.
            dropout_perc (float, optional):     Probability of an element to be zeroed.
            nonlinearity (torch activation, optional):       pytorch.nn activation fuction to use. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            tisc_initialization (int or str, optional):      Initialization argument defining TiSc parameter initialization settings. See :py:func:`~toast.heat.timescaleL.TSC_input.__init__` for more information.
            tisc_scale_multiplier (int, optional):           Exponential base designating how fast the window size increases.
            path_save_progress (str, optional): Path to the parent directory where progress and tracking data should be saved.
            description (str, optional):        A description describing the contents/structure of the network for easy reference.
            comment (str, optional):            A comment about what this network is being used to do/test/investigate for later identification of tests run.

        Returns:
            An instance of the PerceptronNN class representing a deep neural network.
        '''
        # nonlinearity = nn.Tanh

        self.decision_threshold  = 0.5
        self.one_hot_output = False

        # Define the expected layer types, used for organizing potential debugging output.
        layer_type_dict = {'TiScLayer':   [tsl.TSC_input],
                           'TiScTrLayer': [tsl.TSCTranspose_input],
                           'NormLayer':   [nn.BatchNorm1d],
                           'NonlinLayer': [nn.Tanhshrink, nonlinearity]} # Dropout layer type not identified, will be categorized as "Other"
        # Call Network __init__() to initialize standard network variables and functions
        super(TiscEncDecN, self).__init__( layer_type_dict=layer_type_dict, path_save_progress=path_save_progress, description=description, description_short='tiscencdec', comment=comment)
        self.is_regenerator = True

        # Define the linear layers
        tisc_enc_channels = []
        tisc_dec_channels = []
        for _ in range(num_tisc_channels):
            layers_enc_temp = []
            layers_dec_temp = []
            tisc_enc_temp = tsl.TSC_input(          layer_size[0][0], layer_size[0][1], scale_multiplier=tisc_scale_multiplier, bias=True, initialization=tisc_initialization)
            tisc_dec_temp = tsl.TSCTranspose_input( layer_size[0][0], layer_size[0][1], scale_multiplier=tisc_scale_multiplier, bias=True, initialization=tisc_initialization)
            layers_enc_temp.append(tisc_enc_temp)
            layers_dec_temp.append(tisc_dec_temp)

            if nonlinearity is not None:
                layers_enc_temp.append(nonlinearity())
            if isinstance( dropout, list) and (len(dropout) > 0) and dropout[0]:
                layers_enc_temp.append(nn.Dropout(p=dropout_perc))
            elif isinstance( dropout, bool) and dropout:
                layers_enc_temp.append(nn.Dropout(p=dropout_perc))
            tisc_enc_channels.append( nn.Sequential(*layers_enc_temp))
            tisc_dec_channels.append( nn.Sequential(*layers_dec_temp))
        
        self.layers_encode = nn.Sequential( *tisc_enc_channels)
        self.layers_decode = nn.Sequential( *tisc_dec_channels)

        self.new_layers_added()
        

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.

        Args:
            data (tensor): Data to be processed by the network.

        Returns:
            The output when passing the input data through the network
        '''

        tisc_output = torch.zeros_like(data, device=data.device, requires_grad=True)
        for channel_enc, channel_dec in zip(self.layers_encode, self.layers_decode):
            tisc_output = tisc_output.add( channel_dec( channel_enc(data)))
            
        return tisc_output


    def convert_to_decision( self, output):
        '''This method overwrites the forward method of the Network class, in case special behavior is desired
        Convertrs a network output to a definite decision based on thresholds highest likelihood values.

        Args:
            output: Value output by network

        Returns:
            Float Tensor representing the final decision of the network.
        '''
        return None



if __name__ == "__main__":
    testNet = TiscMlpN( [[4, 16384], 300, 100, 50, 20, 1], num_tisc_input=3, dropout=[True, True, True, True, False])
    
    print('Network:')
    print(testNet)

    print('Fin.')

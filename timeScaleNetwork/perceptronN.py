'''
    A Class that contains all the functionality and tracking necessary to train and debug Deep Neural Networks.
'''
from timeScaleNetwork import network as toast_network
import torch.nn as nn
import torch



class PerceptronN(toast_network.Network):

    def __init__(self, layer_size, dropout=[], dropout_perc=0.2, batchnorm=[], nonlinearity=nn.ReLU, nonlinearity_final=nn.Sigmoid, path_save_progress='./', description='', comment='none'):
        '''Creates a multilayer perceptron network object.

        Args:
            layer_size (1D iterable):           Sequential list of the size of each layer, starting with the input layer and ending with the output layer.
            dropout (1D iterable, optional):    Iterable of Boolean Values: Sequentially decide if the correlating layer in layer_size iterable should be followed by a dropout layer. If value(s) don't exist they will be set to False.
            dropout_perc (float, optional):     Probability of an element to be zeroed.
            nonlinearity (torch activation, optional):       pytorch.nn activation fuction to use. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            nonlinearity_final (torch activation, optional): pytorch.nn activation fuction to use after the final layer. Pass a reference to the class, not an instance of the class (don't use parenthesis).
            path_save_progress (str, optional): Path to the parent directory where progress and tracking data should be saved.
            description (str, optional):        A description describing the contents/structure of the network for easy reference.
            comment (str, optional):            A comment about what this network is being used to do/test/investigate for later identification of tests run.

        Returns:
            An instance of the PerceptronN class representing a deep neural network.
        '''
        # nonlinearity = nn.Tanh

        self.decision_threshold  = 0.5
        self.one_hot_output = True if (layer_size[-1] > 1) else False

        # Define the expected layer types, used for organizing potential debugging output.
        layer_type_dict = {'LinrLayer':   [nn.Linear], 
                           'NormLayer':   [nn.BatchNorm1d],
                           'NonlinLayer': [nonlinearity, nonlinearity_final]} # Dropout layer type not identified, will be categorized as "Other"
        # Call Network __init__() to initialize standard network variables and functions
        super(PerceptronN, self).__init__( layer_type_dict=layer_type_dict, path_save_progress=path_save_progress, description=description, description_short='mlp', comment=comment)

        # Define the linear layers
        self.layers = define_sequential_linear_layers( layer_size, dropout=dropout, dropout_perc=dropout_perc, batchnorm=batchnorm, nonlinearity=nonlinearity, nonlinearity_final=nonlinearity_final)

        self.new_layers_added()
        

    def forward( self, data):
        '''Overwrites the forward method of the Network class to process data based on our network structure.

        Args:
            data (tensor): Data to be processed by the network.

        Returns:
            The output when passing the input data through the network
        '''
        return self.layers( data)


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


def define_sequential_linear_layers( layer_size, dropout=[], batchnorm=[], dropout_perc=0.2, nonlinearity=nn.ReLU, nonlinearity_final=nn.Sigmoid):
    '''Streamlined function to generate sequential object containting standard layers in an order commonly seen in perceptron architectures.

    It is expected for most users to use the more comprehensive :py:class:`PerceptronN` class when defining perceptron networks, however 
    this function is spearated out for integration with other class definitions to maintain consistency within the toast framework.

    Args:
        layer_size (1D iterable):           Sequential list of the size of each layer, starting with the input layer and ending with the output layer.
        dropout (1D iterable, optional):    Iterable of Boolean Values: Sequentially decide if the correlating layer in layer_size iterable should be followed by a dropout layer. If value(s) don't exist they will be set to False.
        dropout_perc (float, optional):     Probability of an element to be zeroed.
        nonlinearity (torch activation, optional):       pytorch.nn activation fuction to use. Pass a reference to the class, not an instance of the class (don't use parenthesis).
        nonlinearity_final (torch activation, optional): pytorch.nn activation fuction to use after the final layer. Pass a reference to the class, not an instance of the class (don't use parenthesis).
        
    Returns:
        torch.nn.Sequential object containting the requested linear layers.
    '''
    # if dropout layers are not specified or not enough are specified, fill in with FALSE
    if len(dropout) < len( layer_size)-1:
        dropout = dropout + [False]*(len(layer_size)-len(dropout)-1)
    if len(batchnorm) < len( layer_size):
        batchnorm = batchnorm + [False] * (len(layer_size) - len(batchnorm)-1)

    # Add layers together in order: Layer of requested size -> batchnorm -> nonlinearity -> dropout -> repeat
    # Complete for all but the last layer
    sequential_layers = []
    for i in range( len(layer_size)-2):
        sequential_layers.append( nn.Linear( layer_size[i], layer_size[i+1]))
        if batchnorm[i]:
            sequential_layers.append(nn.BatchNorm1d(layer_size[i+1]))
        if nonlinearity is not None:
            sequential_layers.append( nonlinearity())
        if dropout[i]:
            sequential_layers.append(nn.Dropout(p=dropout_perc))

    # For the last layer, skip the dropout layer and use the defined final nonlinearity
    sequential_layers.append( nn.Linear( layer_size[-2], layer_size[-1]))
    if nonlinearity_final is not None:
        sequential_layers.append( nonlinearity_final())

    # Officially define the layers and layer types
    return nn.Sequential( *sequential_layers)


if __name__ == "__main__":
    testNet = PerceptronN( [23*5*256, 300, 100, 50, 20, 1], dropout=[True, True, True, True, False])
    
    print('Network:')
    print(testNet)

    print('Fin.')

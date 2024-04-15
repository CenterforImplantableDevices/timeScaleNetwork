from numpy import max as np_max, min as np_min
from numpy import mean, std, percentile
from os.path import join, exists
from datetime import datetime
from os import remove, mkdir
from copy import deepcopy
from sys import maxsize
import pickle
import time
import torch.optim as t_optim
import torch.nn as nn
import torch



class Network(nn.Module):
    '''
        A Class that contains all the functionality and tracking necessary to train and debug Deep Neural Networks.
    '''

    def __init__(self, layer_type_dict={}, path_save_progress='./', description='', description_short='toast', comment='none'):
        '''Initialization function to create an instance of data.

        Args:  
            layer_type_dict (dict, optional):   Utility attribute to declare the expected layer types of the network. Helps to organize debugging/tracking functoinality
            path_save_progress (str, optional): Path to the parent directory where progress and tracking data should be saved.
            description (str, optional):        Description describing the contents/structure of the network for easy reference.
            description_short (str, optional):  Short few-character description describing the network type, to be inserted into filenames.
            comment (str, optional):            Comment about what this network is being used to do/test/investigate for later identification of tests run.

        Returns:
            An instance of the Class class representing a deep neural network.
        '''
        super(Network, self).__init__()

        # Officially define the layers
        self.layer_type_dict = layer_type_dict
        
        self.registered_forward_hooks   = []
        self.registered_backward_hooks  = []
        self.track_activation_setup     = False
        self.track_activationGrad_setup = False

        self.device = "cpu"

        # Debugging and tracking attributes
        self.debug = False
        self.loss_best = float( maxsize)
        self.epochs                 = 0
        self.accuracy_hidden        = -1
        self.falsepos_hidden        = -1
        self.falseneg_hidden        = -1
        self.loss_train_history     = []
        self.loss_test_history      = []
        self.accuracy_train_history = []
        self.accuracy_test_history  = []
        self.accuracy_train_falsepos_history = []
        self.accuracy_train_falseneg_history = []
        self.accuracy_test_falsepos_history  = []
        self.accuracy_test_falseneg_history  = []
        self.new_layers_added()

        # Variables important for network identification and saving
        ts = datetime.now()
        self.creation_ts       = "{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}".format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
        self.description       = description
        self.description_short = description_short
        self.comment           = comment
        self.save_path_prefix  = path_save_progress
        self.update_save_paths(create_folders=False)

        self.is_regenerator = False


    def forward( self, data):
        '''This method intended to be overwritten by child class.  Default activity is defined below
        Overriding the nn.Module forward function to process data based on our network structure.

        Args:
            data: Data to be processed by the network.

        Returns:
            The output after passing ``data`` through the network.
        '''
        print('\n\nWARNING: REMEMBER TO OVERWRITE THE forward FUNCTION\n')
        return 0


    def convert_to_decision( self, output):
        '''This method intended to be overwritten by child class.  Default activity is defined below
        Converts a network output to a definite decision based on thresholds highest likelihood values.

        Args:
            output: Value output by network

        Returns:
            Float Tensor representing the final decision of the network.
        '''

        print('\n\nWARNING: REMEMBER TO OVERWRITE THE convert_to_decision FUNCTION\n')
        return (output > 0.5).float()


    def new_layers_added(self):
        '''Resets all the layer-specific tracking attributes of the network object to match new layer dimensions. Can also be used to reset tracking variables if needed.
        '''
        self.weights_test_avg          = { p: -1 for p in self.parameters()}
        self.weights_test_std          = { p: -1 for p in self.parameters()}
        self.weights_test_q1           = { p: -1 for p in self.parameters()}
        self.weights_test_q3           = { p: -1 for p in self.parameters()}
        self.weights_test_min          = { p: -1 for p in self.parameters()}
        self.weights_test_max          = { p: -1 for p in self.parameters()}
        self.weights_test_avg_history  = { p: [] for p in self.parameters()}
        self.weights_test_std_history  = { p: [] for p in self.parameters()}
        self.weights_test_q1_history   = { p: [] for p in self.parameters()}
        self.weights_test_q3_history   = { p: [] for p in self.parameters()}
        self.weights_test_min_history  = { p: [] for p in self.parameters()}
        self.weights_test_max_history  = { p: [] for p in self.parameters()}
        self.activations_test_avg         = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_std         = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_q1          = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_q3          = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_min         = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_max         = { l: -1 for l in self.get_list_of_layers()}
        self.activations_test_avg_history = { l: [] for l in self.get_list_of_layers()}
        self.activations_test_std_history = { l: [] for l in self.get_list_of_layers()}
        self.activations_test_q1_history  = { l: [] for l in self.get_list_of_layers()}
        self.activations_test_q3_history  = { l: [] for l in self.get_list_of_layers()}
        self.activations_test_min_history = { l: [] for l in self.get_list_of_layers()}
        self.activations_test_max_history = { l: [] for l in self.get_list_of_layers()}
        self.gradCAM_latest           = None
        self.gradCAM_expanded_latest  = None
        self.gradCAM_alpha_latest     = None


    def train_with( self, dataloader_to_train, learning_rate=0.05, momentum=0.8, weight_decay=0, loss_fn=nn.MSELoss(), optimizer='SGD', save_progress=False, track_accuracy=False, keep_history=True, show_timings=False):
        '''Performs one epoch of training, while also tracking/logging data that has been set to be tracked/logged.

        Args:  
            dataloader_to_train:                   Dataloader that inherits from torch.utils.data.Dataset to be used during training. Typically a Dataset_Bread instance from a Data_Bread.training class attribute
            learning_rate (float, optional):       Learning rate to use during training. Typical value is 1e-3
            momentum (float, optional):            Momentum coefficient.  Typical value is on the order of 0.8
            weight_decay (float, optional):        Weight decay coefficient.  Typical value is on the order of 1e-2
            loss_fn (torch.nn loss fcn, optional): Torch.nn derived loss function to use when calculating loss
            optimizer (str, optional):             Optimizer function to use. Current supported optimizers are: sgd, rmsprop, adam, adagrad, asgd, rprop, adadelta, adamw, adamax.
            save_progress (bool, optional):        Whether to save the network after the epoch. Will save the network and also replace the "best" network file if performance is better. 
            track_accuracy (bool, optional):       Whether to track the accuracy during training. Typically false.
            keep_history (bool, optional):          Whether to append tracked variables to the network history
            show_timings (bool, optional):         Whether to show the timings of each step performed during training. Used for debug purposes.

        Returns:
            No return value. Updates the parameters/weights of the network and append any tracked variables, as requested.
        '''

        # Define the optimizer function
        if isinstance( optimizer, str):
            if optimizer.lower() == 'sgd':
                optimizer_fn = t_optim.SGD(      filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer.lower() == 'rmsprop':
                optimizer_fn = t_optim.RMSprop(  filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer.lower() == 'adam':
                optimizer_fn = t_optim.Adam(     filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'adagrad':
                optimizer_fn = t_optim.Adagrad(  filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'asgd':
                optimizer_fn = t_optim.ASGD(     filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'rprop':
                optimizer_fn = t_optim.Rprop(    filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
            elif optimizer.lower() == 'adadelta':
                optimizer_fn = t_optim.Adadelta( filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'adamw':
                optimizer_fn = t_optim.AdamW(    filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'adamax':
                optimizer_fn = t_optim.Adamax(   filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer.lower() == 'lbfgs':
                optimizer_fn = t_optim.LBFGS(    filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        elif issubclass( type(optimizer), t_optim.Optimizer):
            optimizer_fn = optimizer
        else:
            print('\n\nWARNING: Optimizer ' + optimizer + ' not recognized.  Using defauld SGD optimizer')
            optimizer_fn = t_optim.SGD( filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # Define/reset Tracking Variables
        self.loss_train = 0
        if track_accuracy:
            number_samples      = 0
            self.accuracy_train = 0
            self.accuracy_train_falsepos = 0
            self.accuracy_train_falseneg = 0
        # Set network to train mode, which will turn on dropout layers, track the gradients/computation graphs, etc
        self.train()

        # Train for one full iteration of the samples provided
        if show_timings:
            start_time = time.time()
        for samples, labels in dataloader_to_train:
            if show_timings:
                print('\t\tTime to get data:\t{:>9.3f} s'.format(time.time() - start_time))
                start_time = time.time()

            samples, labels = samples.to(self.device), labels.to(self.device)
            if show_timings:
                print('\t\tTime to move to GPU:\t{:>9.3f} s'.format(time.time() - start_time))
                start_time = time.time()

            optimizer_fn.zero_grad()
            outputs = self.forward( samples)
            if show_timings:
                print('\t\tTime to run forward:\t{:>9.3f} s'.format(time.time() - start_time))
                start_time = time.time()

            if self.is_regenerator:
                loss = loss_fn(outputs, samples)
            else:
                loss = loss_fn( outputs, labels)

            # Track variables that are requested
            if track_accuracy:
                number_samples += len(samples)
                label_compare = self.convert_to_decision(outputs) - self.convert_to_decision( labels)
                self.accuracy_train          += torch.sum(label_compare == 0).item()
                self.accuracy_train_falsepos += torch.sum(label_compare  > 0).item()
                self.accuracy_train_falseneg += torch.sum(label_compare  < 0).item()

            if self.debug:
                number_layers = sum([1 for temp in self.parameters()])
                print('----------')
                for num, layer in enumerate( self.parameters()):
                    if (num == 0) or (num == number_layers-2):
                        print('Layer ', num, ': ', end='')
                        print(layer)
                print('Sampels: ', torch.reshape( samples, (-1,)))
                print('labels:  ', torch.reshape( labels, (-1,)))
                print('Outputs: ', torch.reshape( outputs, (-1,)))
                print('Loss:    ', torch.reshape( loss, (-1,)))
                # input('Press [Enter] to continue...')

            loss.backward()
            optimizer_fn.step()
            if show_timings:
                print('\t\tTime to run backward:\t{:>9.3f} s'.format(time.time() - start_time))
                start_time = time.time()

            self.loss_train += loss.item()
            if show_timings:
                start_time = time.time()

        # Finish calculating tracked variables
        self.loss_train /= len(dataloader_to_train.dataset)
        if keep_history:
            self.loss_train_history.append( deepcopy(self.loss_train))
        if track_accuracy:
            self.accuracy_train          /= number_samples
            self.accuracy_train_falsepos /= number_samples
            self.accuracy_train_falseneg /= number_samples
            if keep_history:
                self.accuracy_train_history.append(          deepcopy(self.accuracy_train))
                self.accuracy_train_falsepos_history.append( deepcopy(self.accuracy_train_falsepos))
                self.accuracy_train_falseneg_history.append( deepcopy(self.accuracy_train_falseneg))

        self.epochs += 1

        if save_progress:
            self.update_save_paths()
            self.save_checkpoint( self.path_chpt)
        if (self.loss_train < self.loss_best):
            self.loss_best = self.loss_train
            if save_progress:
                self.update_save_paths()
                self.save_checkpoint( self.path_best)
    

    def test_with( self, dataloader_to_test, loss_fn=nn.MSELoss(), track_accuracy=False, track_weights=False, track_activations=False, track_activationsGrad=False, keep_history=True, show_timings=False):
        '''Tests network performance through one iteration of the given dataset, while also tracking/logging data that has been set to be tracked/logged.

        Args:  
            dataloader_to_test:                     Dataloader that inherits from torch.utils.data.Dataset to be used during testing. Typically a Dataset_Bread instance from a Data_Bread.testing class attribute
            loss_fn (torch.nn loss fcn, optional):  Torch.nn derived loss function to use when calculating loss
            track_accuracy (bool, optional):        Whether to track the accuracy during training. Typically false.
            track_weights (bool, optional):         Whether to log the the weights during training. Typically false.
            track_activations (bool, optional):     Whether to track activation values during forward propogation of network. Typically false.
            track_activationsGrad (bool, optional): Whether to track activation gradient values from backpropogation. Necessary for some visualizatoin features, like GradCAM. Typically false.
            keep_history (bool, optional):          Whether to append tracked variables to the network history
            show_timings (bool, optional):          Whether to show the timings of each step performed during training. Used for debug purposes.

        Returns:
            No return value.  Will append any tracked variables, as requested.
        '''
        # Define/reset Tracking Variables
        self.loss_test = 0
        if track_accuracy:
            number_samples_tested = 0
            self.accuracy_test    = 0
            self.accuracy_test_falsepos = 0
            self.accuracy_test_falseneg = 0
        if track_activations:
            if not self.track_activation_setup:
                print('\n\nWARNING: TRACK ACTIVATIONS NOT SETUP. SKIPPING TRACKING DURING TEST')
                track_activations = False
            else:
                self.layer_activations = {}
                self.layer_activations_keys_ordered = []
        if track_activationsGrad:
            if not self.track_activationGrad_setup:
                print('\n\nWARNING: TRACK ACTIVATIONS NOT SETUP. SKIPPING TRACKING DURING TEST')
                track_activationsGrad = False
            else:
                self.layer_activationsGrad = {}
                self.layer_activationsGrad_raw = None
                self.layer_activationsGrad_keys_ordered = []

        # Set network to eval mode, which will turn off dropout layers, stop tracking the gradients/computation graphs, etc for faster performance
        self.eval()
        with torch.no_grad():
            # Test one full iteration of the samples provided
            if show_timings:
                start_time = time.time()

            for samples, labels in dataloader_to_test:
                if show_timings:
                    print('\t\tTime to get data:\t{:>9.3f} s'.format(time.time() - start_time))
                    start_time = time.time()

                samples, labels = samples.to(self.device), labels.to(self.device)
                if show_timings:
                    print('\t\tTime to move to GPU:\t{:>9.3f} s'.format(time.time() - start_time))
                    start_time = time.time()

                outputs = self.forward( samples)
                if show_timings:
                    print('\t\tTime to run forward:\t{:>9.3f} s'.format(time.time() - start_time))
                    start_time = time.time()

                # Track the variables that are requested
                if track_accuracy:
                    number_samples_tested += len(samples)
                    label_compare = self.convert_to_decision(outputs) - self.convert_to_decision( labels)
                    self.accuracy_test          += torch.sum(label_compare == 0).item()
                    self.accuracy_test_falsepos += torch.sum(label_compare  > 0).item()
                    self.accuracy_test_falseneg += torch.sum(label_compare  < 0).item()

                if self.is_regenerator:
                    loss = loss_fn( outputs, samples)
                else:
                    loss = loss_fn( outputs, labels)
                self.loss_test += loss.item()

                if show_timings:
                    start_time = time.time()
                
        # Finish calculating tracked variables
        self.loss_test /= len(dataloader_to_test.dataset)
        if keep_history:
            self.loss_test_history.append( deepcopy(self.loss_test))
        if track_accuracy:
            self.accuracy_test          /= number_samples_tested
            self.accuracy_test_falsepos /= number_samples_tested
            self.accuracy_test_falseneg /= number_samples_tested
            if keep_history:
                self.accuracy_test_history.append(          deepcopy(self.accuracy_test))
                self.accuracy_test_falsepos_history.append( deepcopy(self.accuracy_test_falsepos))
                self.accuracy_test_falseneg_history.append( deepcopy(self.accuracy_test_falseneg))
        
        if track_weights:
            for layer_param in self.parameters():
                raw_weight_value = layer_param.data.cpu().numpy()
                # raw_weight_value = deepcopy( layer_weights.data).cpu().numpy() # If GPU memory becomes an issue, use this line
                self.weights_test_avg[layer_param] = mean(raw_weight_value)
                self.weights_test_std[layer_param] = std( raw_weight_value)
                self.weights_test_q1[ layer_param] = percentile( raw_weight_value, 25)
                self.weights_test_q3[ layer_param] = percentile( raw_weight_value, 75)
                self.weights_test_min[layer_param] = np_min( raw_weight_value)
                self.weights_test_max[layer_param] = np_max( raw_weight_value)
                if keep_history:
                    self.weights_test_avg_history[layer_param].append( deepcopy(self.weights_test_avg[layer_param]))
                    self.weights_test_std_history[layer_param].append( deepcopy(self.weights_test_std[layer_param]))
                    self.weights_test_q1_history[ layer_param].append( deepcopy(self.weights_test_q1[ layer_param]))
                    self.weights_test_q3_history[ layer_param].append( deepcopy(self.weights_test_q3[ layer_param]))
                    self.weights_test_min_history[layer_param].append( deepcopy(self.weights_test_min[layer_param]))
                    self.weights_test_max_history[layer_param].append( deepcopy(self.weights_test_max[layer_param]))
        
        if track_activations:
            for layer_key, layer_activations in self.layer_activations.items():
                raw_activations_value = layer_activations.data.cpu().numpy()
                self.activations_test_avg[layer_key] = mean(raw_activations_value)
                self.activations_test_std[layer_key] = std( raw_activations_value)
                self.activations_test_q1[ layer_key] = percentile( raw_activations_value, 25)
                self.activations_test_q3[ layer_key] = percentile( raw_activations_value, 75)
                self.activations_test_min[layer_key] = np_min( raw_activations_value)
                self.activations_test_max[layer_key] = np_max( raw_activations_value)
                if keep_history:
                    self.activations_test_avg_history[layer_key].append( deepcopy(self.activations_test_avg[layer_key]))
                    self.activations_test_std_history[layer_key].append( deepcopy(self.activations_test_std[layer_key]))
                    self.activations_test_q1_history[ layer_key].append( deepcopy(self.activations_test_q1[ layer_key]))
                    self.activations_test_q3_history[ layer_key].append( deepcopy(self.activations_test_q3[ layer_key]))
                    self.activations_test_min_history[layer_key].append( deepcopy(self.activations_test_min[layer_key]))
                    self.activations_test_max_history[layer_key].append( deepcopy(self.activations_test_max[layer_key]))

        
    def get_labeled_outputs_from( self, dataloader_to_test):
        '''Will return true labels and corresponding output values from a dataloader for reference

        Args:
            dataloader_to_test: Dataloader that inherits from torch.utils.data.Dataset to be used to find labels & outputs. Typically a Dataset_Bread instance from a Data_Bread.testing class attribute

        Returns:
            labels (numpy array), outputs (numpy array):  **labels**: Numpy array of labels used during training.  Note if one_hot encoding was used, one_hot labels will be returned.
            **outputs**: Numpy array of the values output from the network
        '''
        all_labels  = torch.Tensor()
        all_outputs = torch.Tensor()

        # Set network to eval mode, which will turn off dropout layers, stp[ tracking the gradients/computation graphs, etc for faster performance
        self.eval()
        with torch.no_grad():
            for samples, labels in dataloader_to_test:
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.forward( samples)

                all_labels  = torch.cat(( all_labels,  labels.cpu()))
                all_outputs = torch.cat((all_outputs, outputs.cpu()))
                
        return all_labels.numpy(), all_outputs.numpy()


    def save_checkpoint( self, path=''):
        '''Saves a "checkpoint" of the network parameters/state so that it can be loaded in case of failure during training.
        Note that history/tracked variables are not saved or loaded. This is a reduced file-size saving method that only saves what is necessary to checkpoint performance.

        Args:
            path (str, optional): Path (including filename) to save the network to

        Returns:
            No return value. Network will be saved to a new file (or overwrite existing file of same name)
        '''

        if len(path) == 0:
            self.update_save_paths()
            path = self.path_network
        with open( path + '.pkl', 'wb') as f:
            pickle.dump( self.state_dict(), f)


    def load_checkpoint( self, path=None):
        '''Loads a network from a saved "checkpoint", if saved using above function.
        Note that history/tracked variables are not saved or loaded. "checkpoint"s are a reduced file-size saving method that only saves what is necessary to checkpoint performance.

        Args:
            path (str, optional): path (including filename) to a pickle file containing a saved network.

        Returns:
            Returns 1 if successful. Saved parameters/attributes will be loaded into the network.
        '''
        try:
            with open( path+'.pkl', 'rb') as f:
                state_dict = pickle.load(f)
                self.load_state_dict( state_dict)
            return 1
        except Exception as e:
            print('ERROR loading model from ', path)
            raise(e)
    

    def save_network( self, path=None, remove_checkpoint=True):
        '''Saves the entire network instance for later reference.  All values, parameters, and attributes are pickled saved.

        Args:
            path (str, optional):               Path (including filename) to save the network to
            remove_checkpoint (bool, optional): Whehter to remove existing checkpoints

        Returns:
            No return value. Network will be saved to a new file (or overwrite existing file of same name)
        '''
        self.remove_all_hooks() # Some hooks are unpickleable
        self.update_save_paths()
        if not path:
            path = self.path_network
        with open( path+'.pkl', 'wb') as f:
            pickle.dump( self, f)
        if remove_checkpoint:
            remove( self.path_chpt+'.pkl')
        return 1
    

    def update_save_paths(self, create_folders=True):
        '''Updates the network attributes representing save paths. Used when attributes are edited after initialization that are used in the save_path definitions.

        Args:
            create_folders (bool, optional): Whether to create any mising folders for saved networks.

        Returns:
            No return value. All path attributes will be updated.
        '''

        dirname_checkpoints = 'network_train_checkpoint'
        if create_folders and not( exists( join(self.save_path_prefix, dirname_checkpoints))):
            mkdir( join(self.save_path_prefix, dirname_checkpoints))
        self.path_chpt = join( self.save_path_prefix, dirname_checkpoints, self.description_short + '_checkpoint_' + self.description + '_' + self.creation_ts + '_' + self.comment)
        self.path_best = join( self.save_path_prefix, dirname_checkpoints, self.description_short + '_best_' + self.description + '_' + self.creation_ts + '_' + self.comment)
        dirname_trained = 'network_trained'
        if create_folders and not( exists( join(self.save_path_prefix, dirname_trained))):
            mkdir( join(self.save_path_prefix, dirname_trained))
        self.path_network = join( self.save_path_prefix, dirname_trained, self.description_short + '_network_' + self.description + '_' + self.creation_ts + '_' + self.comment)


    def get_list_of_layers(self):
        '''Returns an ordered list of the layer objects used by the network. These describe the network topology/organization used by the network.

        Args:
            No Arguments.

        Returns:
            An ordered list of the layer objects found. Note ``nn.Sequential`` objects are ignored in this search.
        '''

        identified_layers = []
        # Define a function that will be used recursively
        def find_and_add_layers( network):
            '''
            FIXME: Use this function instead:
                        def modules(self) -> Iterator['Module']:
                            r"""Returns an iterator over all modules in the network.
                            Yields:
                                Module: a module in the network
                            Note:
                                Duplicate modules are returned only once. In the following
                                example, ``l`` will be returned only once.
                            Example::
                                >>> l = nn.Linear(2, 2)
                                >>> net = nn.Sequential(l, l)
                                >>> for idx, m in enumerate(net.modules()):
                                        print(idx, '->', m)
                                0 -> Sequential(
                                (0): Linear(in_features=2, out_features=2, bias=True)
                                (1): Linear(in_features=2, out_features=2, bias=True)
                                )
                                1 -> Linear(in_features=2, out_features=2, bias=True)
                            """
                            for name, module in self.named_modules():
                                yield module

            '''
            # Iterate through all the attached modules add them to the identified_layers list
            for _, layer in network._modules.items():
                # Do not add sequential layers, but iterate within them
                if isinstance( layer, nn.Sequential):
                    find_and_add_layers( layer)
                else:
                    identified_layers.append( layer)
        
        # Call the recursive function on self network object
        find_and_add_layers( self)
        return identified_layers


    def get_param_modules(self):
        '''Returns an ordered list of the parameter modules contained in the network. These describe the weight values contained by the network.
            
        Note:
            Linear layers contain a weights parameter module and a bias parameter module. (1 layer, 2 parameter modules)
            
            Non-linearity layers do not contain parameters as they only perform a fixed operation.

        Args:
            No Arguments.

        Returns:
            An ordered list of the parameter modules found.
        '''

        identified_modules = []
        # Define a function that will be used recursively
        def find_and_add_modules( input_module):
            '''Returns an ordered list of the module parameter objects used by the network. These describe the parameters used by the network.

            Args:
                input_module (module): Network or smaller module object

            Returns:
                An ordered list of the module parameters
            '''
            for _, module in input_module._modules.items():
                # Recurse through a module object until it is None so that we know nothing else is contained or embedded in submodules
                if module is None:
                    continue
                find_and_add_modules( module)
                '''Internal Function used to recursively append modules to identified_modules variable
                FIXME: Use this function instead: 

                            de.f parameters(self, recurse: bool = True) -> Iterator[Parameter]:
                                r"""Returns an iterator over module parameters.
                                This is typically passed to an optimizer.
                                Args:
                                    recurse (bool): if True, then yields parameters of this module
                                        and all submodules. Otherwise, yields only parameters that
                                        are direct members of this module.
                                Yields:
                                    Parameter: module parameter
                                Example::
                                    >>> for param in model.parameters():
                                    >>>     print(type(param), param.size())
                                    <class 'torch.Tensor'> (20L,)
                                    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
                                """
                                for name, param in self.named_parameters(recurse=recurse):
                                    yield param
                '''
                # Once a module has no submodules, add all its parameters
                for _ in module._parameters.items():
                    identified_modules.append( module)
        
        # Recurse through a module object until it is None so that we know nothing else is contained or embedded in submodules
        find_and_add_modules( self)
        return identified_modules


    def remove_forward_hooks( self):
        '''Unregisters all forward hooks.

        May be necessary to remove hooks to move network to/from GPU or to save network, as some implemented hooks are unpickleable.

        Args:
            No Arguments.

        Returns:
            No Return Value. Hooks will be unregistered.
        '''
        self.track_activation_setup = False
        for hook in self.registered_forward_hooks:
            hook.remove()


    def remove_backward_hooks( self):
        '''Unregisters all backwards hooks. 
        
        May be necessary to remove hooks to move network to/from GPU or to save network, as some implemented hooks are unpickleable.

        Args:
            No Arguments.

        Returns:
            No Return Value. Hooks will be unregistered.
        '''
        for hook in self.registered_backward_hooks:
            hook.remove()


    def remove_all_hooks( self):
        '''Unregisters all forward and backwards hooks. 

        May be necessary to remove hooks to move network to/from GPU or to save network, as some implemented hooks are unpickleable.

        Args:
            No Arguments.

        Returns:
            No Return Value. Hooks will be unregistered.
        '''
        self.remove_forward_hooks()
        self.remove_backward_hooks()


    def _add_forwardHook_to_layers( self, network, hook_function):
        '''Internal function used to add a passed hook_function to layers. 
        
        Users should not use this function, it only exists due to recursion/class structure nuances.
        '''
        # Iterate through all the attached modules register the hook function with that layer
        for _, layer in network._modules.items():
            if isinstance( layer, nn.Sequential):
                self._add_forwardHook_to_layers( layer, hook_function)
            else:
                handle = layer.register_forward_hook( hook_function)
                self.registered_forward_hooks.append( handle)


    def _add_backwardHook_to_layers( self, network, hook_function):
        '''Internal function used to add a passed hook_function to layers. 
        
        Users should not use this function, it only exists due to recursion/class structure nuances.
        '''
        # Iterate through all the attached modules register the hook function with that layer
        for _, layer in network._modules.items():
            if isinstance( layer, nn.Sequential):
                self._add_backwardHook_to_layers( layer, hook_function)
            else:
                handle = layer.register_full_backward_hook( hook_function)
                self.registered_backward_hooks.append( handle)


    def add_hook_activation( self):
        '''Single comprehensive function that adds hooks/attributes necessary to track activation values of the network during training/testing.

        Args:
            No Arguments.

        Returns:
            No return value. Hooks will be registered and attributes initialized.
        '''
        self.track_activation_setup = True
        self.layer_activations = {}
        self.layer_activations_keys_ordered = []

        def hook_fn( module, input, output):
            ''' Define the function that the hook will execute with the standard hook input variables
                To track activatoins, we create a dictionary with the layer objects as keys, and the output activation values as values
                Also assemble an ordered list of the activation layer keys for organized iteration later
            '''
            if not self.training:
                # key = id(module)
                key = module
                # Note: Based on https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor , 
                # torch.empty_like(output).copy_(output).cpu() is fastest way to copy tensor to cpu
                try:
                    self.layer_activations[key] = torch.cat((self.layer_activations[key], torch.empty_like(output).copy_(output).cpu()), dim=0)
                except KeyError:
                    self.layer_activations[key] = torch.empty_like(output).copy_(output).cpu()
                    if key not in self.layer_activations_keys_ordered:
                        self.layer_activations_keys_ordered.append(key)
                except Exception as e:
                    print('\n\nWARNING: HOOK ERROR\n')
                    raise e

        self._add_forwardHook_to_layers( self, hook_fn)
    

    def add_hook_activationGrad( self, layerTarget=None):
        '''Single comprehensive function that adds hooks/attributes necessary to track activation values of the network during training/testing.

        Args:
            layerTarget (nn.Module layer, optional): Single layer to track activation gradients for. If not specified, all layers are tracked. Testing of this specific optinal input was not comprehensive, may need tweaking to assure proper cleanup and workflow with other functions.

        Returns:
            No return value. Hooks will be registered and attributes initialized.
        '''
        self.track_activationGrad_setup = True
        self.layer_activationsGrad = {}
        self.layer_activationsGrad_raw = None
        self.layer_activationsGrad_keys_ordered = []
        
        def hook_fn_backward( module, grad_input, grad_output):
            ''' Define the function that the hook will execute with the standard hook input variables
                To track activations, we create a dictionary with the layer objects as keys, and the output activation values as values
                Also assemble an ordered list of the activation layer keys for organized iteration later
            '''
            if not self.training:
                # key = id(module)
                key = module
                # Note: Based on https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor , 
                # torch.empty_like(output).copy_(output).cpu() is fastest way to copy tensor to cpu
                try:
                    self.layer_activationsGrad[key] = torch.cat((self.layer_activationsGrad[key], torch.empty_like(grad_output[0]).copy_(grad_output[0]).cpu()), dim=0)
                except KeyError:
                    self.layer_activationsGrad[key] = torch.empty_like(grad_output[0]).copy_(grad_output[0]).cpu()
                    if key not in self.layer_activationsGrad_keys_ordered:
                        self.layer_activationsGrad_keys_ordered.append(key)
                except Exception as e:
                    print('\n\nWARNING: HOOK ERROR\n')
                    raise e
        
        # Define a forward hook which will add the backwards hook onto the activation tensor, once it is created
        # This 2-hook setup is necessary because we have to wait until output is generated to add a hook
        def hook_fn_forward_tensor( module, input, output):
            ''' Define a forward hook which will add the backwards hook onto the activation tensor, once it is created
                This 2-hook setup is necessary because we have to wait until output is generated to add a hook
            '''
            if not self.training:
                handle_t = output.register_hook(hook_fn_backward_tensor)
                self.registered_backward_hooks.append(handle_t)

        def hook_fn_backward_tensor( grad):
            ''' Define a backward hook which will keep track of the activation gradients
                This 2-hook setup is necessary because we have to wait until output is generated to add a hook
            '''
            if not self.training:
                # Note: Based on https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor , 
                # torch.empty_like(output).copy_(output).cpu() is fastest way to copy tensor to cpu
                try:
                    self.layer_activationsGrad_raw = torch.cat((self.layer_activationsGrad_raw, torch.empty_like(grad).copy_(grad).cpu()), dim=0)
                except TypeError:
                    self.layer_activationsGrad_raw = torch.empty_like(grad).copy_(grad).cpu()
                except Exception as e:
                    print('\n\nWARNING: HOOK ERROR\n')
                    raise e
        

        if layerTarget is not None:
            layerTarget.register_forward_hook( hook_fn_forward_tensor)
            # handle = layerTarget.register_backward_hook(hook_fn_backward)
            # self.registered_backward_hooks.append(handle)
        else:
            self._add_backwardHook_to_layers( self, hook_fn_backward)


    def track_activations_cleanup( self):
        '''Helper function to easily cleanup all activation-tracking varaibles/hooks after training.

        May be necessary to move network to/from GPU or to save network, as the activation tracking implemention is unpickleable.

        Args:
            No Arguments.

        Returns:
            No return value. Hooks will be unregistered and attributes cleared.
        '''
        self.track_activation_setup     = False
        self.track_activationGrad_setup = False
        self.remove_all_hooks()
        
        self.layer_activations = {}
        self.layer_activations_keys_ordered = []
        self.layer_activationsGrad = {}
        self.layer_activationsGrad_raw = None
        self.layer_activationsGrad_keys_ordered = []

import numpy as np
from ._generic_model import GenericModel

###############################################################################

def correct_dimensions(s, targetlength):
    """
    Parameters
    ----------
    s: None, scalar or 1D array
    
    targetlength: int
        Expected length of s.
 
    Returns
    ----------                
    : None if s is None, else numpy vector of length targetlength
    """ 
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            # If there is a single value, this command creates
            # a vector of the required size filled with that value.
            s = np.array([s] * targetlength)                                       
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s

###############################################################################

def linear(x):
    return x    

###############################################################################

class ESN (GenericModel):
    """
    
    Echo State Networks.
    
    For details on the technique, see the paper 
    by Lemos et al. (2021) - Echo State Network Based Soft Sensor 
    for Monitoring and Fault Detection of Industrial Processes, 
    https://doi.org/10.1016/j.compchemeng.2021.107512

    This code has been modified and adapted from the following repository:
    https://github.com/cknd/pyESN"

    Parameters
    ----------
    n_reservoir: int, optional
        Number of neurons in the reservoir.
    spectral_radius: float, optional
        Spectral radius of the recurrent weight matrix.
    sparsity: float, optional
        Proportion of recurrent weights set to zero.
    noise: float, optional
        Noise added to each neuron (regularization).
    input_shift: float or numpy.array
        Scalar or vector of length n_inputs to be added to each
        input dimension before feeding it to the network.
    input_scaling: float or numpy.array
        Scalar or vector of length n_inputs to be multiplied
        with each input dimension before feeding it to the network.
    teacher_forcing: boolean, optional
        If True, results in an ESN with output layer recursion to the 
        dynamic reservoir.
    teacher_scaling: float, optional 
        Factor applied to the target signal.
    teacher_shift: float, optional
        Additive term applied to the target signal.
    out_activation: func, optional
        Output activation function (applied to the readout).
    inverse_out_activation: func, optional
        Inverse of the output activation function.
    random_state: int or np.rand.RandomState, optional
        Positive integer seed, np.rand.RandomState object,
        or None to use numpy's builting RandomState.
    silent: boolean, optional
        Suppress messages.
    """    
    
    ###########################################################################
    
    def __init__(self, n_reservoir=400,
                 spectral_radius=0.95, sparsity=0.95, noise=0.01, 
                 input_shift=None,
                 input_scaling=None, teacher_forcing=False, 
                 feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=linear, inverse_out_activation=linear,
                 random_state=None, silent=True):

        
        self.has_Y = True
        
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = input_shift
        self.input_scaling = input_scaling

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        #self.initweights()
        
        self.name = 'ESN'
        
    ###########################################################################
        
    def initweights(self):       
        """
        Initializes the weights of the network.
        """    
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1
  
    ###########################################################################
        
    def _update(self, state, input_pattern, output_pattern):
        """
        Executes one update step,
        i.e., calculates the next state of the network by applying the
        recurrent weights to the last state and feeding in the current input
        and output patterns.

        Parameters
        ----------    
        state: numpy.array
            Last state.
        input_pattern: numpy.array
            Input pattern.        
        state: numpy.array
            Last state.              
        """
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))
        return (np.tanh(preactivation)
              + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))
  
    ###########################################################################
        
    def _scale_inputs(self, inputs):
        """
        For each input dimension j: multiply by the j'th entry in
        the input_scaling argument and, subsequently, add the j'th entry
        of the input_shift argument.
        
        Parameters
        ----------    
        inputs: numpy.array
            Inputs.          
        Returns
        ----------    
        inputs: numpy.array
            Inputs in the new scale.        
        """
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs
  
    ###########################################################################
        
    def _scale_teacher(self, teacher):
        """
        Multiply the teacher/target signal by the teacher_scaling argument,
        then add the teacher_shift argument.
        
        Parameters
        ----------    
        teacher: numpy.array
            Teacher signal.          
        Returns
        ----------    
        teacher: numpy.array
            Teacher signal in the new scale.          
        """
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher
  
    ###########################################################################
        
    def _unscale_teacher(self, teacher_scaled):
        """Inverse operation of the _scale_teacher method.
        
        Parameters
        ----------    
        teacher_scaled: numpy.array
            Teacher signal in the new scale.          
        Returns
        ----------    
        teacher_scaled: numpy.array
            Original teacher signal.          
        """
        
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled
  
    ###########################################################################
    
    def pre_train (self, *args, **kwargs):
        
        super().pre_train (*args, **kwargs)
        
        self.n_inputs = np.shape(self.X_train)[1]
        self.n_outputs = np.shape(self.Y_train)[1]
        self.input_shift = correct_dimensions(self.input_shift, 
                                              self.n_inputs)
        self.input_scaling = correct_dimensions(self.input_scaling, 
                                                self.n_inputs)
        self.initweights()
    
    ###########################################################################

    def train_core (self):
        """
        Harvest the network's reaction to training data,
        train the readout weights.
        The result is the output of the network on the training data,
        using the trained weights.
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if self.X_train.ndim < 2:
            self.X_train = np.reshape(self.X_train, (len(self.X_train), -1))
        if self.Y_train.ndim < 2:
            self.Y_train = np.reshape(self.Y_train, (len(self.Y_train), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(self.X_train.values)
        teachers_scaled = self._scale_teacher(self.Y_train.values)

        if not self.silent:
            print("harvesting states...")
            print("")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((self.X_train.shape[0], self.n_reservoir))
        for n in range(1, self.X_train.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
            print("")
        # we'll disregard the first few states:
        transient = min(int(self.X_train.shape[1] / 10), 100)
        # include the raw inputs:
        extended_states = np.hstack((states, inputs_scaled))
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                self.inverse_out_activation(teachers_scaled[transient:, :])).T
        
        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = self.X_train.values[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # apply learned weights to the collected states:
        #pred_train = self._unscale_teacher(self.out_activation(
        #    np.dot(extended_states, self.W_out.T)))
        
    ###########################################################################
        
    def map_from_X(self, X, continuation=True):
        """
        Apply the learned weights to the network's reactions to new inputs.

        Parameters
        ----------  
        inputs: numpy.array 
            Inputs of shape (N_test_samples x n_inputs)
        continuation: boolean, optional
            If True, start the network from the last training state.
        Returns
        ----------  
        : numpy.array
            Output activation matrix.
        """
        if X.ndim < 2:
            X = np.reshape(X, (len(X), -1))
        n_samples = X.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        X = np.vstack([lastinput, self._scale_inputs(X)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[
                n + 1, :] = self._update(states[n, :], 
                                         X[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                              np.concatenate([states[n + 1, :], X[n + 1, :]])))
       
        return self._unscale_teacher(self.out_activation(outputs[1:]))

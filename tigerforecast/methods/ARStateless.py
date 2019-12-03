"""
LSTM neural network method
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import tigerforecast
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import *
from tigerforecast.utils.optimizers.losses import *
import pickle


class ARStateless(Method):
    """
    Description: Produces outputs from a randomly initialized seq2seq LSTM neural network.
                 Supposed to be used in batch seq2seq mode. Not online mode. 
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, n=1, m=1, l = 32, optimizer = None):
        """
        Description: Randomly initialize the Stateless AR.
        Args:
            n (int): Observation/output dimension.
            m (int): Input action dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of LSTM.
            optimizer (instance of Optimizer Class): optimizer choice
            lr (float): learning rate for update
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.l = n, m, l

        # initialize parameters
        glorot_init = stax.glorot() # returns a function that initializes weights
        W_lnm = glorot_init(generate_key(), (l, m, n)) # maps l inputs to output
        b = np.zeros((m, 1)) # bias 
        self.params = [W_lnm, b]
        self.x = np.zeros((l, m))


        """ private helper methods"""

        @jax.jit
        def _predict(params, x):
            y = np.einsum('ijk,ik->j', params[0], x) + params[1] 
            return y

        self.transform = lambda x: float(x) if (self.m == 1) else x
        self._predict = jax.vmap(_predict, in_axes=(None, 0))
        if optimizer==None:
            optimizer_instance = OGD(loss=batched_mse)
            self._store_optimizer(optimizer_instance, self._predict)
        else:
            self._store_optimizer(optimizer, self._predict)

    def initialize_with_ckpt(self, n=1, m=1, l = 32, optimizer = None, filename=None):
        if filename==None:
            print("initialize_with_ckpt should be called with a filename. Use initalize instead")
            raise
        else:
            self.initialize(n=n,m=m,l=l,optimizer=optimizer)
            self.load(filename)


    def _check_format(self, x):

        if x.ndim < 3:
            print("x needs to be shaped as [batch_size, sequence_length, n]")
            raise
        elif x.shape[1]!=self.l:
            print("The second dimension of x should be of size l")
            raise
        elif x.shape[2]!=self.n:
            print("The third dimension of x should be of size n")
            raise
    
    def predict(self, x):
        """
        Description: Predict next value given observation
        Args:
            x (numpy.ndarray): Observation. 1st Dimension is batch_size.
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized
        self._check_format(x)
        self.x = x
        return self._predict(self.params, x)
    
    def forecast(self, x, timeline = 1):
        ### TODO: See if this function needs to be implemented. 
        raise NotImplementedError
      

    def update(self, y):
        """
        Description: Updates parameters
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        assert self.initialized
        self.params = self.optimizer.update(self.params, self.x, y)
        return

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.params, f)
        f.close()
        return

    def load(self, filename):
        """
            TODO: Check for dimensions and filename error
        """
        f = open(filename, 'rb')
        self.params = pickle.load(f)
        f.close()
        return 
     

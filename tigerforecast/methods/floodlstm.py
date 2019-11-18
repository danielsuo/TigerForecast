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


class FloodLSTM(Method):
    """
    Description: Produces outputs from a randomly initialized seq2seq LSTM neural network.
                 Supposed to be used in batch seq2seq mode. Not online mode. 
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, n=1, m=1, l = 32, h = 100, e_dim = 10, num_sites = 1000, optimizer = None):
        """
        Description: Randomly initialize the LSTM.
        Args:
            n (int): Observation dimension.
            m (int): Output dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of LSTM.
            optimizer (instance of Optimizer Class): optimizer choice
            lr (float): learning rate for update
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.l, self.h, self.e_dim, self.num_sites = n, m, l, h, e_dim, num_sites
        self.postembed_n = self.n - 1 + self.e_dim 
        # initialize parameters
        glorot_init = stax.glorot() # returns a function that initializes weights
        W_hh = glorot_init(generate_key(), (4*h, h)) # maps h_t to gates
        W_xh = glorot_init(generate_key(), (4*h, self.postembed_n)) # maps x_t to gates
        W_out = glorot_init(generate_key(), (m, h)) # maps h_t to output
        W_embed = glorot_init(generate_key(), (num_sites, e_dim))
        b_h = np.zeros(4*h)
        b_h = jax.ops.index_update(b_h, jax.ops.index[h:2*h], np.ones(h)) # forget gate biased initialization
        self.params = [W_hh, W_xh, W_out, W_embed, b_h]
        self.hid = np.zeros(h)
        self.cell = np.zeros(h)


        """ private helper methods"""

        @jax.jit
        def _fast_predict(carry, x):
            params, hid, cell = carry # unroll tuple in carry
            W_hh, W_xh, W_out, W_embed, b_h = params
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
            gate = np.dot(W_hh, hid) + np.dot(W_xh, x) + b_h 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            y = np.dot(W_out, next_hid)
            return (params, next_hid, next_cell), y

        @jax.jit
        def _predict(params, x):
            full_x = np.concatenate((params[3][x[:,0].astype(np.int32),:], x[:,1:]), axis=-1)
            _, y = jax.lax.scan(_fast_predict, (params, np.zeros(h), np.zeros(h)), full_x)
            return y

        self.transform = lambda x: float(x) if (self.m == 1) else x
        self._fast_predict = _fast_predict
        self._predict = jax.jit(jax.vmap(_predict, in_axes=(None, 0)))
        if optimizer==None:
            optimizer_instance = Adam(loss=batched_mse, learning_rate=1.0)
            self._store_optimizer(optimizer_instance, self._predict)
        else:
            self._store_optimizer(optimizer, self._predict)

    def initialize_with_ckpt(self, n=1, m=1, l = 32, h = 64, e_dim = 10, optimizer = None, filename=None):
        if filename==None:
            print("initialize_with_ckpt should be called with a filename. Use initalize instead")
            raise
        else:
            self.initialize(n=n,m=m,l=l,h=h,optimizer=optimizer)
            self.load(filename)

    def _process_x(self, x):
        if x.ndim < 3:
            print("x needs to be shaped as [batch_size, sequence_length, n]")
            raise
        elif x.shape[1]!=self.l:
            print("The second dimension of x should be of size l")
            raise
        elif x.shape[2]!=self.n:
            print("The third dimension of x should be of size n")
            raise

        self.x = x       
    
    def predict(self, x):
        """
        Description: Predict next value given observation
        Args:
            x (numpy.ndarray): Observation. 1st Dimension is batch_size.
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized
        self._process_x(x)
        return self._predict(self.params, self.x)
    
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
     

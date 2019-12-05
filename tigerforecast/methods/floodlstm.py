"""
LSTM neural network method
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import jax.random as random
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

    def initialize(self, n=1, m=1, l = 32, h = 100, 
                   e_dim = 10, num_sites = 1000, optimizer = None, dp_rate=0., filename=None
                  ):
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
        self.params = {'W_hh' : W_hh,
                       'W_xh' : W_xh,
                       'W_out' : W_out,
                       'W_embed': W_embed,
                       'b_h' : b_h}
        self.hid = np.zeros(h)
        self.cell = np.zeros(h)
        self.dp_rate = dp_rate
        self.keep_rate = 1.0 - dp_rate
        if filename != None:
            self.load(filename)

        """ private helper methods"""

        @jax.jit
        def _fast_predict(carry, x):
            params, hid, cell, dp_masks, t = carry # unroll tuple in carry
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
            # x *= dp_masks['input_masks']
            # x = np.where(dp_masks['input_masks'], x / self.dp_rate, 0)
            # hid *= dp_masks['recurrent_masks'][t]
            if self.dp_rate > 0:
                hid = np.where(dp_masks['recurrent_masks'][t], hid / self.keep_rate, 0)
            gate = np.dot(params['W_hh'], hid) + np.dot(params['W_xh'], x) + params['b_h'] 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            # next_hid *= dp_masks['output_masks'][t]
            if self.dp_rate > 0:
                next_hid = np.where(dp_masks['output_masks'][t], next_hid / self.keep_rate, 0)
            y = np.dot(params['W_out'], next_hid)
            return (params, next_hid, next_cell, dp_masks, t+1), y

        @jax.jit
        def _predict(params, x):
            full_x = np.concatenate((params['W_embed'][x[:,0].astype(np.int32),:], x[:,1:]), axis=-1)
            if self.dp_rate > 0:
                full_x = np.where(self.dp_masks['input_masks'], full_x / self.keep_rate, 0)
            _, y = jax.lax.scan(_fast_predict, (params, np.zeros(h), np.zeros(h), self.dp_masks, 0), full_x)
            return y

        self.transform = lambda x: float(x) if (self.m == 1) else x
        self._fast_predict = _fast_predict
        self._predict = jax.jit(jax.vmap(_predict, in_axes=(None, 0)))
        if optimizer==None:
            # last_loss = lambda x,y : np.mean( ( x[:,-1,:]-y[:,-1,:] )**2 )
            optimizer_instance = OGD(loss=batched_mse, learning_rate=0.01)
            self._store_optimizer(optimizer_instance, self._predict)
        else:
            self._store_optimizer(optimizer, self._predict)

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
        self.dp_masks = None
        if self.dp_rate > 0:
            self.dp_masks = self.generate_dp_masks(x, self.keep_rate)
            # self.x = np.where(self.dp_masks['input_masks'], x / self.keep_rate, 0)
        
        return self._predict(self.params, self.x)

    def generate_dp_masks(self, x, rate):
        batch_size = x.shape[0]
        dp_masks = {}
        input_masks = random.bernoulli(generate_key(), rate, (self.postembed_n,))
        recurrent_masks = random.bernoulli(generate_key(), rate, (self.l, self.h))
        output_masks = random.bernoulli(generate_key(), rate, (self.l, self.h))
        dp_masks['input_masks'] = input_masks
        dp_masks['recurrent_masks'] = recurrent_masks
        dp_masks['output_masks'] = output_masks
        return dp_masks
    
    def forecast(self, x, timeline = 1):
        ### TODO: See if this function needs to be implemented. 
        raise NotImplementedError
      

    def update(self, y, dynamic=False):
        """
        Description: Updates parameters
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        assert self.initialized
        self.new_params = self.optimizer.update(self.params, self.x, y)
        # self.new_params = self.optimizer.update(self.params, x, y)
        if dynamic:
            prior_lambda = 0.6
            prior_step = {key: self.initial_params[key] - self.params[key] for key in self.params}
            
            for key in self.params:
                if key == 'W_embed': # skip training the embedding entirely
                    continue
                self.params[key] = self.new_params[key] + prior_lambda * prior_step[key]
        else:
            self.params = self.new_params
        return


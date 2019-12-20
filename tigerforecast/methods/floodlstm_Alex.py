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

        W_batchnorm_x = np.ones((self.l, 4*h))
        b_batchnorm_x = np.zeros((self.l, 4*h))
        W_batchnorm_hid = np.ones((self.l, 4*h))
        b_batchnorm_hid = np.zeros((self.l, 4*h))
        W_batchnorm_cell = np.ones((self.l, h))
        b_batchnorm_cell = np.zeros((self.l, h))

        b_h = np.zeros(4*h)
        b_h = jax.ops.index_update(b_h, jax.ops.index[h:2*h], np.ones(h)) # forget gate biased initialization
        self.params = {'W_hh' : W_hh,
                       'W_xh' : W_xh,
                       'W_out' : W_out,
                       'W_embed': W_embed,
                       'b_h' : b_h,
                       'W_batchnorm_x' : W_batchnorm_x,
                       'b_batchnorm_x' : b_batchnorm_x,
                       'W_batchnorm_hid' : W_batchnorm_hid,
                       'b_batchnorm_hid' : b_batchnorm_hid,
                       'W_batchnorm_cell' : W_batchnorm_cell,
                       'b_batchnorm_cell' : b_batchnorm_cell}

        self.hid = np.zeros(h)
        self.cell = np.zeros(h)
        self.dp_rate = dp_rate
        self.keep_rate = 1.0 - dp_rate
        self.running_mean = np.zeros(self.postembed_n) 
        self.running_var = np.zeros(self.postembed_n)
        if filename != None:
            self.load(filename)

        """ private helper methods"""

        @jax.jit
        def gate_products_mapper(params, x, hid):
            W_hh_hid = np.dot(params['W_hh'], hid)
            W_xh_x = np.dot(params['W_xh'], x)
            return (W_hh_hid, W_xh_x)

        def _BN_mapper(params, z, mean, var, t, suffix):
            return params['b_batchnorm_' + suffix][t] + params['W_batchnorm_' + suffix][t] * (z - mean)/((var + 1e-5)**0.5)

        BN_mapper = jax.jit(_BN_mapper, static_argnums=[5])
        # BN_mapper = _BN_mapper

        @jax.jit
        def gate_mapper(params, bn_W_hh_hid, bn_W_xh_x):
            return bn_W_hh_hid + bn_W_xh_x + params['b_h']

        @jax.jit
        def cell_mapper(gate, cell):
            sigmoid = lambda x: 1. / (1. + np.exp(-x))
            i, f, g, o = np.split(gate, 4)
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            return (next_cell, o)

        @jax.jit
        def hid_mapper(next_cell, o):
            sigmoid = lambda x: 1. / (1. + np.exp(-x))
            next_hid = sigmoid(o) + np.tanh(next_cell)
            return next_hid

        @jax.jit
        def y_mapper(params, next_hid):
            return np.dot(params['W_out'], next_hid)

        # x.shape = (batch, time, dim)
        def scanner(carry, x_t):
            params, hid, cell, t = carry
            (W_hh_hid, W_xh_x) = jax.vmap(gate_products_mapper, in_axes=(None, 0, 0))(params, x_t, hid) # W_hh_hid.shape = (batch, 4*h)
            # print("x_t.shape = " + str(x_t.shape))
            # print("hid.shape = " + str(hid.shape))
            # print("W_hh_hid.shape = " + str(W_hh_hid.shape))
            # print("W_xh_x.shape = " + str(W_xh_x.shape))
            mean_W_hh_hid, var_W_hh_hid = np.mean(W_hh_hid, axis=0), np.var(W_hh_hid, axis=0)
            mean_W_xh_x, var_W_xh_x = np.mean(W_xh_x, axis=0), np.var(W_xh_x, axis=0)
            # print("mean_W_hh_hid.shape = " + str(mean_W_hh_hid.shape))
            # print("var_W_hh_hid.shape = " + str(var_W_hh_hid.shape))
            
            BN_W_hh_hid = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, W_hh_hid, mean_W_hh_hid, var_W_hh_hid, t, 'hid')
            BN_W_xh_x = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, W_xh_x, mean_W_xh_x, var_W_xh_x, t, 'x')
            # print("BN_W_hh_hid.shape = " + str(BN_W_hh_hid.shape))
            # print("BN_W_xh_x.shape = " + str(BN_W_xh_x.shape))


            gate = jax.vmap(gate_mapper, in_axes=(None, 0, 0))(params, BN_W_hh_hid, BN_W_xh_x) # gate.shape = (batch, 4*h)
            (next_cell, o) = jax.vmap(cell_mapper, in_axes=(0, 0))(gate, cell)
            mean_cell, var_cell = np.mean(np.mean(next_cell, axis=0)), np.var(next_cell, axis=0)
            BN_next_cell = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, next_cell, mean_cell, var_cell, t, 'cell')
            next_hid = jax.vmap(hid_mapper, in_axes=(0, 0))(BN_next_cell, o)

            # hid, cell = next_hid, next_cell
            y = jax.vmap(y_mapper, in_axes=(None, 0))(params, next_hid)
            # print("y = " + str(y[0][0]))
            # print("y.shape = " + str(y.shape))
            # ys = jax.ops.index_update(ys, jax.ops.index[:,t,0], np.squeeze(y))
            return (params, next_hid, next_cell, t+1), y

        def batchnorm_predict(params, x):
            full_x = np.concatenate((params['W_embed'][x[:,:,0].astype(np.int32),:], x[:,:,1:]), axis=-1)
            batch_size = x.shape[0]
            hid = np.zeros((batch_size,h))
            cell = np.zeros((batch_size,h))
            ys = np.zeros((batch_size, self.l, 1))
            
            full_x_T = np.transpose(full_x, axes=[1,0,2])
            # print(full_x_T)
            _, y = jax.lax.scan(scanner, (params, hid, cell, 0), full_x_T)
            # y = np.transpose(y, axes=[1,0,2])
            # print(y)
            return np.transpose(y, axes=[1,0,2])

            '''
            for t in range(self.l):
                (W_hh_hid, W_xh_x) = jax.vmap(gate_products_mapper, in_axes=(None, 0, 0, None))(params, full_x, hid, t) # W_hh_hid.shape = (batch, 4*h)
                mean_W_hh_hid, var_W_hh_hid = np.mean(W_hh_hid, axis=0), np.var(W_hh_hid, axis=0)
                mean_W_xh_x, var_W_xh_x = np.mean(W_xh_x, axis=0), np.var(W_xh_x, axis=0)
                
                BN_W_hh_hid = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, W_hh_hid, mean_W_hh_hid, var_W_hh_hid, t, 'hid')
                BN_W_xh_x = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, W_xh_x, mean_W_xh_x, var_W_xh_x, t, 'x')
                
                gate = jax.vmap(gate_mapper, in_axes=(None, 0, 0))(params, BN_W_hh_hid, BN_W_xh_x) # gate.shape = (batch, 4*h)
                (next_cell, o) = jax.vmap(cell_mapper, in_axes=(0, 0))(gate, cell)
                mean_cell, var_cell = np.mean(np.mean(next_cell, axis=0)), np.var(next_cell, axis=0)
                BN_next_cell = jax.vmap(BN_mapper, in_axes=(None, 0, None, None, None, None))(params, next_cell, mean_cell, var_cell, t, 'cell')
                next_hid = jax.vmap(hid_mapper, in_axes=(0, 0))(BN_next_cell, o)

                hid, cell = next_hid, next_cell
                y = jax.vmap(y_mapper, in_axes=(None, 0))(params, hid)
                # print("y.shape = " + str(y.shape))
                ys = jax.ops.index_update(ys, jax.ops.index[:,t,0], np.squeeze(y))
            return ys'''

        self._batchnorm_predict = batchnorm_predict

        @jax.jit
        def _fast_predict(carry, x):
            params, hid, cell = carry # unroll tuple in carry
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
            gate = np.dot(params['W_hh'], hid) + np.dot(params['W_xh'], x) + params['b_h'] 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            y = np.dot(params['W_out'], next_hid)
            return (params, next_hid, next_cell), y

        @jax.jit
        def _fast_predict_with_dropout(carry, x):
            params, hid, cell, recurrent_mask, output_mask, t = carry # unroll tuple in carry
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?

            hid *= recurrent_mask[t]
            
            gate = np.dot(params['W_hh'], hid) + np.dot(params['W_xh'], x) + params['b_h'] 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)

            y = np.dot(params['W_out'], next_hid * output_mask[t])

            return (params, next_hid, next_cell, recurrent_mask, output_mask, t+1), y

        @jax.jit
        def _predict(params, x):
            full_x = np.concatenate((params['W_embed'][x[:,0].astype(np.int32),:], x[:,1:]), axis=-1)
            _, y = jax.lax.scan(_fast_predict, (params, np.zeros(h), np.zeros(h)), full_x)
            return y

        @jax.jit
        def _predict_with_dropout(params, x):
            x, input_mask, recurrent_mask, output_mask = x
            full_x = np.concatenate((params['W_embed'][x[:,0].astype(np.int32),:], x[:,1:]), axis=-1) * input_mask
            _, y = jax.lax.scan(_fast_predict_with_dropout, (params, np.zeros(h), np.zeros(h), recurrent_mask, output_mask, 0), full_x)
            return y

        self.transform = lambda x: float(x) if (self.m == 1) else x
       
        self._fast_predict = _fast_predict
        self._predict = jax.jit(jax.vmap(_predict, in_axes=(None, 0)))

        self._fast_predict_with_dropout = _fast_predict_with_dropout
        self._predict_with_dropout = jax.jit(jax.vmap(_predict_with_dropout, in_axes=(None, 0)))

        if optimizer==None:
            optimizer = OGD(loss=batched_mse, learning_rate=0.01)

        if self.dp_rate > 0.:
            pred_fn = self._predict_with_dropout
        else:
            pred_fn = self._batchnorm_predict
            # pred_fn = self._predict

        self._store_optimizer(optimizer, pred_fn)

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
    
    def predict(self, x, inference=False):
        """
        Description: Predict next value given observation
        Args:
            x (numpy.ndarray): Observation. 1st Dimension is batch_size.
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized
        self._process_x(x)
        '''
        W_xh_mapper = lambda x_seq: np.dot(self.params['W_xh'], x_seq)
        W_xh_x = jax.vmap(W_xh_mapper, in_axes=0)(x)
        mean_W_xh_x = np.mean(W_xh_x, axis=0)
        var_W_xh_x = np.var(W_xh_x, axis=0)'''

        if self.dp_rate > 0 and not inference:
            self.input_masks, self.recurrent_masks, self.output_masks = self.generate_dp_masks(x, self.keep_rate)
            return self._predict_with_dropout(self.params, (self.x, self.input_masks, self.recurrent_masks, self.output_masks))
        else:
            return self._batchnorm_predict(self.params, self.x)
            # return self._predict(self.params, self.x)

    def generate_dp_masks(self, x, keep_rate):
        batch_size = x.shape[0]

        input_masks = random.bernoulli(generate_key(), keep_rate, (batch_size, self.postembed_n,)) / keep_rate
        recurrent_masks = random.bernoulli(generate_key(), keep_rate, (batch_size, self.l, self.h)) / keep_rate
        output_masks = random.bernoulli(generate_key(), keep_rate, (batch_size, self.l, self.h)) / keep_rate

        return input_masks, recurrent_masks, output_masks
    
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
        if self.dp_rate > 0:
            x = (self.x, self.input_masks, self.recurrent_masks, self.output_masks)
        else:
            x = self.x
        self.new_params = self.optimizer.update(self.params, x, y)
        self.params = self.new_params
        return


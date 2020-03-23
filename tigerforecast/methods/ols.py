"""
Stateless AR method
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


class OLS(Method):
    """
    Description: Produces outputs from a randomly initialized seq2seq LSTM neural network.
                 Supposed to be used in batch seq2seq mode. Not online mode. 
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, n=1, l = 32):
        """
        Description: Randomly initialize the Online Least Squares.
        TODO: I am keeping it single class for now 
        Args:
            n (int): Observation/output dimension.
        """
        self.T = 0
        self.initialized = True
        self.n = n

        # initialize parameters
        glorot_init = stax.glorot() # returns a function that initializes weights
        # W_lnm = glorot_init(generate_key(), (l, m, n)) # maps l inputs to output
        W_ln = np.zeros((l,n))
        b = np.zeros((1, 1)) # bias 
        self.params = {'W_ln': W_ln, 'b': b}
        self.x = np.zeros((l, n))
        self.vec = np.zeros((l*n + 1, 1))

        """ private helper methods"""

        @jax.jit
        def _predict(params, x):
            y = params['b'] + np.sum(params['W_ln']*x) 
            return y

        @jit # partial update step for every matrix in method weights list
        def partial_update(A, Ainv, x, grad):
            A = A + np.outer(x)
            inv_val = Ainv @ x
            Ainv = Ainv - np.outer(inv_val, inv_val) / (1 + x.T @ inv_val)
            final_sol_flat = Ainv @ grad
            final_sol_W = np.reshape(final_sol_flat[:-1], (l,n))
            final_sol_b = np.array([[final_sol_flat[-1]]])
            return A, Ainv, final_sol_W, final_sol_b
        self.partial_update = partial_update

        self.transform = lambda x: float(x) if (self.m == 1) else x
        self._predict = jax.vmap(_predict, in_axes=(None, 0))
        
    def initialize_with_ckpt(self, n=1, l = 32, filename=None):
        if filename==None:
            print("initialize_with_ckpt should be called with a filename. Use initalize instead")
            raise
        else:
            self.initialize(n=n,l=l)
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
        x = np.squeeze(x)
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
        ### The entire OLS Logic goes here
        ravelled_x = np.append(np.ravel(self.x), [1.])
        self.vec += y*ravelled_x

        # initialize A
        if self.A is None:
            self.A = np.eye(self.l*self.n+1) * self.eps
            self.Ainv = np.eye(self.l*self.n+1) * (1 / self.eps)

        # partial_update automatically reshapes flat_grad into correct params shape
        self.A, self.Ainv, final_sol_W, final_sol_b = self.partial_update(self.A, self.Ainv, ravelled_x, self.vec) 
        self.params['W_ln'] = final_sol_W
        self.params['b'] = final_sol_b
        return

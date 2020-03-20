'''
RMSProp optimizer
'''

from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
from jax import jit, grad
import jax.numpy as np

class RMSProp(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, include_x_loss= False, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = {'reg':0.0, 'beta_2': 0.999, 'eps': 1e-8, 'clip_norm':True, 'max_norm':1.0}
        self.hyperparameters.update(hyperparameters)
        for key, value in self.hyperparameters.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.v = None

        self.pred = pred
        self.loss = loss
        self.include_x_loss = include_x_loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

        @jit # helper update method
        def _update(params, grad, v, max_norm):
            scale = 1.0
            if self.clip_norm:
                scale = np.maximum(self.max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad.values()]))
            grad = {k : dw/scale for (k,dw) in grad.items()}
            v_t = {k:self.beta_2 * v[k] + (1. - self.beta_2) * np.square(grad[k]) for k in v.keys()}
            lr = self.lr
            new_params = {k: (w - lr * grad[k] / (np.sqrt(v_t[k]) + self.eps)) for k, w in params.items()}
            return new_params, new_v, max_norm
        self._update = _update

    def reset(self): # reset internal parameters
        self.v = None

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of method pred method
            x (float): input to method
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized
        assert type(params) == dict, "optimizers can only take params in dictionary format"
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class

        # Make everything a list for generality
        if self.v == None: # first run
            self.v = {k:np.zeros(dw.shape) for k, dw in grad.items()}

        updated_params = self._update(params, grad, self.v, self.max_norm)
        new_params, self.v, self.max_norm = updated_params
        return new_params

    def __str__(self):
        return "<RMSProp Optimizer, lr={}>".format(self.lr)




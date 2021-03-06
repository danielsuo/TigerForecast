'''
AdaGrad optimizer
'''
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
from jax import jit, grad
import jax.numpy as np


class Adagrad(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = {'max_norm':True, 'reg': 0.0}
        self.hyperparameters.update(hyperparameters)
        for key, value in self.hyperparameters.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.G = None
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

        @jit
        def _update(params, grad, G, max_norm):
            new_G = {k:(g + np.square(dw)) for (k, g), dw in zip(G.items(), grad.values())}
            max_norm = np.where(max_norm, np.maximum(max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad.values()])), max_norm)
            lr = self.lr / np.where(max_norm, max_norm, 1.)
            new_params = {k:(w - lr * dw / np.sqrt(g)) for (k, w), dw, g in zip(params.items(), grad.values(), new_G.values())}
            return new_params, new_G, max_norm
        self._update = _update

    def reset(self): # reset internal parameters (self.G)
        self.G = None

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
        if self.G == None: # first run
            self.G = {k: 1e-3 * np.ones(shape=g.shape) for k, g in grad.items()}

        new_params, self.G, self.max_norm = self._update(params, grad, self.G, self.max_norm)
        return new_params

    def __str__(self):
        return "<AdaGrad Optimizer, lr={}>".format(self.lr)



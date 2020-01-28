'''
Dynamic evaluation meta-optimizer.
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error

class DynamicEval(Optimizer):
    """
    Description: Dynamic evaluation meta-optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        optimizer (Optimizer): underlying optimizer to add dynamic evaluation
        prior_weight (float): how much to move towards initialization
        exclude (list or set): parameter names to freeze
    Returns:
        None
    """
    def __init__(self, optimizer=None, prior_weight=0.0, exclude=None):
        self.initialized = False

        self.optimizer = optimizer
        self.prior_weight = prior_weight

        if exclude is None:
            exclude = []
        self.exclude = set(exclude)

        # inherit pointers from base optimizer
        self.pred = optimizer.pred
        self.loss = optimizer.loss

        if self._is_valid_pred(self.pred, raise_error=False) and self._is_valid_loss(self.loss, raise_error=False):
            self.set_predict(self.pred, loss=self.loss)

    def reset(self): # reset internal parameters
        self.optimizer.reset()

    def set_prior(self, params):
        # freeze global prior state
        self.prior = {key: param.copy() for key, param in params.items()}

    def set_predict(self, pred, loss=None):
        super().set_predict(pred, loss)
        self.optimizer.set_predict(pred, loss)

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

        self.next_params = self.optimizer.update(params, x, y, loss=loss)
        final_step = {key: self.next_params[key] + self.prior_weight*(self.prior[key] - params[key])
            for key in params if key not in self.exclude}
        for key in self.exclude:
            final_step[key] = self.prior[key]

        return final_step

    def __str__(self):
        return "<DynamicEval Optimizer, optimizer={}>".format(str(self.optimizer))




# Method class
# Author: John Hallman

from tigerforecast import error
from tigerforecast.methods.optimizers import Optimizer

# class for implementing algorithms with enforced modularity
class Method(object):

    def initialize(self, **kwargs):
        # initializes method parameters
        raise NotImplementedError

    def predict(self, x=None):
        # returns method prediction for given input
        raise NotImplementedError

    def update(self, **kwargs):
        # update parameters according to given loss and update rule
        raise NotImplementedError

    def _store_optimizer(self, optimizer, pred):
        if isinstance(optimizer, Optimizer):
            optimizer.set_predict(pred)
            self.optimizer = optimizer
            return
        if issubclass(optimizer, Optimizer):
            self.optimizer = optimizer(pred=pred)
            return
        raise error.InvalidInput("Optimizer input cannot be stored")

    def help(self):
        # prints information about this class and its methods
        raise NotImplementedError

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __enter__(self):
        return self

# Method class
# Author: John Hallman

from tigerforecast import error
from tigerforecast.utils.optimizers import Optimizer

import pickle

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
            self.optimizer = optimizer()
            self.optimizer.set_predict(pred)
            return
        raise error.InvalidInput("Optimizer input cannot be stored")

    def save(self, filename):
        assert (hasattr(self, "params")), "Model {} does not have params initialized".format(self)
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
        # print(self.params)
        #self.initial_params = [x.copy() for x in self.params]
        f.close()
        return 

    def copy(self):
        # deep copy of params dict
        # usage: A.params = B.copy()
        assert (hasattr(self, "params")), "Model {} does not have params initialized".format(self)
        return {key: param.copy() for key, param in self.params.items()}

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __repr__(self):
        return self.__str__()

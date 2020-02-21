"""
AR(p): Linear combination of previous values
"""

import tigerforecast
import jax
import jax.numpy as np
from tigerforecast.utils.optimizers.losses import mse

class SimpleBoostHet:
    """
    Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False

    def initialize(self, method_id, n = None, m = None, loss=mse, reg=0.0):
        """
        Description: Initializes autoregressive method parameters
        Args:
            method_id (string): list of instances of methods
            method_params (dict): list of dict of params to pass to methods
            N (int): default 3. Number of weak learners
            loss (function): loss function for boosting method
            reg (float): default 1.0. constant for regularization.
        """
        self.initialized = True

        # initialize proxy loss
        def proxy_loss(y_pred, v):
            return np.sum(y_pred*v) + (reg/2) * np.sum(y_pred**2)

        # 1. Maintain N copies of the algorithm 
        assert len(method_id) > 0
        self.N = len(method_id)
        self.methods = []
        '''
        for m_params in method_params:
            m_params['n'] = n
            m_params['m'] = m'''
        for i in range(self.N):
            # new_method = tigerforecast.method(method_id)
            # new_method.initialize(**method_params[i])
            new_method = method_id[i]
            new_method.optimizer.set_loss(proxy_loss) # proxy loss
            self.methods.append(new_method)

        def _prev_predict(x):
            y = []
            cur_y = 0
            for i, method_i in enumerate(self.methods):
                eta_i = 2 / (i + 2)
                y_pred = method_i.predict(x)
                y_pred = np.array([a.flatten() for a in y_pred])
                cur_y = (1 - eta_i) * cur_y + eta_i * y_pred
                y.append(cur_y)
                # print("i = " + str(i))
                # print("y_pred = " + str(y_pred[:3]))
                # print("y_pred.shape = " + str(y_pred.shape))
                # print("cur_y.shape = " + str(cur_y.shape))
                # print("cur_y = " + str(cur_y[:3]))
                
            return [np.zeros(shape=y[0].shape)] + y
        self._prev_predict = _prev_predict
        
        def _get_grads(y_true, prev_predicts):
            g = jax.grad(loss)
            v_list = [g(y_prev, y_true) for y_prev in prev_predicts]
            return v_list
        self._get_grads = jax.jit(_get_grads)


    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return x


    def predict(self, x):
        assert self.initialized
        x = self.to_ndarray(x)
        self._prev_predicts = self._prev_predict(x)
        # print("self._prev_predicts = " + str(self._prev_predicts))
        # print(type(self._prev_predicts[-1]))
        return self._prev_predicts[-1]


    def update(self, y):
        assert self.initialized
        grads = self._get_grads(y, self._prev_predicts)
        for grad_i, method_i in zip(grads[:-1], self.methods):
            method_i.update(grad_i)


    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(AutoRegressor_help)



# string to print when calling help() method
AutoRegressor_help = """

-------------------- *** --------------------

Id: AutoRegressor
Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series

Methods:

    initialize()
        Description:
            Initializes autoregressive method parameters
        Args:
            p (int): Length of history used for prediction

    step(x)
        Description:
            Run one timestep of the method in its environment then update internal parameters
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    update(y, loss, lr)
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): (optional)
            lr (float):
        Returns:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""

import ctsb
import numpy as np 
import matplotlib.pyplot as plt



class ArmaOgd(ctsb.CustomModel):
    """
    Implements the ARMA-OGD algorithm for time series prediction
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self, p = 3):
        """
        Description:
            Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction
        """
        self.initialized = True
        self.past = np.zeros(p + 1)
        self.params = np.zeros(p + 1)
        self.order = p
        self.lr = 1
        self.t = 1
        self.max_norm = 1

    def predict(self, x):
        """
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized

        """ predict"""
        return np.dot(self.params, self.past)

    def update(self, y, loss = None):
        """
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): specifies loss function to be used; defaults to MSE
            lr (float): specifies learning rate; defaults to 0.001.
        Returns:
            None
        """

        """ first update the internal parameters """
        grad = (np.dot(self.params, self.past) - y) * self.past 
        if np.linalg.norm(grad) > self.max_norm:
            self.max_norm = np.linalg.norm(grad)

        self.t = self.t + 1
        self.lr = 1 / (self.max_norm * np.sqrt(self.t))
        self.params = self.params - self.lr * grad 

        """ and then update the indices of the past """
        temp = np.zeros(self.order + 1)
        temp[0:self.order] = self.past[1:self.order + 1]
        temp[self.order] = y
        self.past = temp

        return


    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print('This method implements the ARMA-OGD algorithm. See the paper http://proceedings.mlr.press/v30/Anava13.pdf for details. ')

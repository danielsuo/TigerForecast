"""
S&P 500 daily opening price
"""

import tigerforecast
import os
import jax.numpy as np
import pandas as pd
from math import log # for log returns
from tigerforecast.utils import sp500, get_tigerforecast_dir
from tigerforecast.error import StepOutOfBounds
from tigerforecast.problems.time_series import TimeSeriesProblem

class SP500(TimeSeriesProblem):
    """
    Description: Outputs the daily opening price of the S&P 500 stock market index 
    from January 3, 1986 to June 29, 2018.
    """

    compatibles = set(['SP500-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.data_path = os.path.join(get_tigerforecast_dir(), "data/sp500.csv")
        self.has_regressors = False

    def initialize(self, normalization='log_return'):
        """
        Description: Check if data exists, else download, clean, and setup.
        Args:
            normalization (str/None): if None, no data normalization. if 'log_return', return log(x_t/x_(t-1)).
                if 'return', return (x_t - x_(t-1)) / x_(t-1)
        Returns:
            The first S&P 500 value
        """
        self.initialized = True
        self.normalization = normalization
        if normalization != None:
            assert normalization in ['return', 'log_return'], "normalization must be either None, return, or log_return"
        self.T = 0
        self.df = sp500() # get data
        self.max_T = self.df.shape[0]
        self.has_regressors = False
        self.x_prev = self.df.iloc[self.T, 1]
        return self.df.iloc[self.T, 1]

    def step(self):
        """
        Description: Moves time forward by one day and returns value of the stock index
        Args:
            None
        Returns:
            The next S&P 500 value
        """
        assert self.initialized
        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))

        x_curr = self.df.iloc[self.T, 1]
        if self.normalization == 'return': r = (x_curr - self.x_prev) / self.x_prev
        elif self.normalization == 'log_return': r = log(x_curr / self.x_prev)
        else: r = x_curr
        self.x_prev = x_curr
        return r

    def hidden(self):
        """
        Description: Return the date corresponding to the last value of the S&P 500 that was returned
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized
        return "Timestep: {} out of {}, date: ".format(self.T+1, self.max_T) + self.df.iloc[self.T, 0]

    def close(self):
        """
        Not implemented
        """
        pass

    def __str__(self):
        return "<SP500 Problem>"

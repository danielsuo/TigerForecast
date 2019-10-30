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
from tigerforecast.problems import Problem

class SP500(Problem):
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
        self.has_regressors = False
        self.normalization = normalization
        if normalization != None:
            assert normalization in ['return', 'log_return'], "normalization must be either None, return, or log_return"
        self.T = 0
        df = sp500() # get data
        self.max_T = df.shape[0]
        data = (df['value'].values.tolist())
        if normalization == 'return':
            data = np.array([(data[i+1]-data[i])/data[i] for i in range(len(data)-1)])
            self.std = np.std(data)
            data /= self.std
        elif normalization == 'log_return':
            data = np.array([np.log(data[i+1]/data[i]) for i in range(len(data)-1)])
            self.std = np.std(data)
            data /= self.std
        else:
            data = np.array(data)
            self.std = np.std(data)
        self.data = data
        return self.data[self.T]

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
        return self.data[self.T]


    def __str__(self):
        return "<SP500 Problem>"

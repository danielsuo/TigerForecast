"""
Use FB's Prophet
"""

import numpy as onp
import pandas as pd
import fbprophet as pro
import jax.numpy as np
import datetime
import tigerforecast
from tigerforecast.methods import Method

class Prophet(Method):
    """
    Description: Uses a third part predictor -- FB's Prophet
    """
    compatibles = set(['TimeSeries'])
    
    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self):
        """
        Description: Initialize the method.
        Args:
            None
        Returns:
            None
        """
        self.initialized = True
        self.df, self.t = pd.DataFrame(columns=['ds', 'y']), 0

    def predict(self, x):
        """
        Description: Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        self.m = pro.Prophet()
        c, x = datetime.datetime.fromtimestamp(self.t*1e3), x.copy()[0]
        self.df = self.df.append(pd.Series([c, x], index=self.df.columns), ignore_index=True)
        self.t += 1
        if self.t > 2:
            self.m.fit(self.df)
            df_pred = self.m.make_future_dataframe(periods = 1, include_history = False)
            pred = self.m.predict(df_pred)['yhat']
        else:
            pred = x
        return x

    def update(self, rule=None):
        """
        Description: Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None
        """
        return

    def __str__(self):
        return "<Prophet Method>"


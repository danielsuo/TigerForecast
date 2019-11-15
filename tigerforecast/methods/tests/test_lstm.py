# test the LSTM method class

import tigerforecast
import numpy as onp
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigerforecast.utils import generate_key
from tigerforecast.utils.download_tools import get_tigerforecast_dir
import os
import pandas as pd
import ast

def test_lstm(steps=100, show_plot=True):
    T = steps 
    n, m, l, d = 5, 5, 10, 10
    problem = tigerforecast.problems.LDS_TimeSeries()
    y_true = problem.initialize(n, m, d)
    method = tigerforecast.methods.LSTM()
    method.initialize(m, n, l, d)
    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results = []
    for i in range(T):
        u = random.normal(generate_key(), (n,))
        y_pred = method.predict(u)
        _, y_true = problem.step()
        results.append(loss(y_true, y_pred))
        method.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("LSTM method on LDS_TimeSeries")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_lstm passed")
    return

if __name__=="__main__":
    test_lstm()

# test the LeastSquares method class

import tigerforecast
import jax.numpy as np
import matplotlib.pyplot as plt
from tigerforecast.utils.optimizers import *

def test_least_squares(steps=1000, show_plot=True):
    T = steps 
    problem = tigerforecast.problems.ENSO()
    x, y = problem.initialize(input_signals = ['nino12', 'nino34', 'nino4'])

    method = tigerforecast.methods.LeastSquares()
    method.initialize(x, y, reg = 10.0 * steps)
    loss = lambda y_true, y_pred: np.sum((y_true - y_pred)**2)
 
    results = []

    for i in range(T):
        x, y_true = problem.step()
        y_pred = method.step(x, y_true)
        cur_loss = loss(y_true, y_pred)
        results.append(cur_loss)

    if show_plot:
        plt.plot(results)
        plt.title("LeastSquares method on ARMA problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_least_squares passed")
    return

if __name__=="__main__":
    test_least_squares()
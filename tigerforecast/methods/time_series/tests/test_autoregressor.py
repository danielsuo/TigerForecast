# test the Autogressor method class

import tigerforecast
import jax.numpy as np
import matplotlib.pyplot as plt
from tigerforecast.methods.optimizers import *

def test_autoregressor(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 3
    n = 1
    problem = tigerforecast.problem("ARMA-v0")
    cur_x = problem.initialize(p, q, n = n)

    method = tigerforecast.method("AutoRegressor")
    #method.initialize(p, optimizer = ONS)
    method.initialize(p, optimizer = Adagrad)
    loss = lambda y_true, y_pred: np.sum((y_true - y_pred)**2)
 
    results = []

    for i in range(T):
        cur_y_pred = method.predict(cur_x)
        #print(cur_y_pred.shape)
        #method.forecast(cur_x, 3)
        cur_y_true = problem.step()
        cur_loss = loss(cur_y_true, cur_y_pred)
        method.update(cur_y_true)
        cur_x = cur_y_true
        results.append(cur_loss)

    if show_plot:
        plt.plot(results)
        plt.title("Autoregressive method on ARMA problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_autoregressor passed")
    return

if __name__=="__main__":
    test_autoregressor()
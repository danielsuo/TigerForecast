# test the RNN method class

import tigerforecast
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigerforecast.utils import generate_key

def test_rnn(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 3
    n = 1
    problem = tigerforecast.problem("ARMA-v0")
    y_true = problem.initialize(p=p,q=q, n=1)
    method = tigerforecast.method("RNN")
    method.initialize(n=1, m=1, l=3, h=1)
    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results = []
    for i in range(T):
        u = random.normal(generate_key(), (n,))
        y_pred = method.predict(u)
        y_true = problem.step()
        results.append(loss(y_true, y_pred))
        method.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("RNN method on LDS problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_rnn passed")
    return

if __name__=="__main__":
    test_rnn()
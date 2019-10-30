# test the S&P 500 problem class

import tigerforecast
import jax.numpy as np
import matplotlib.pyplot as plt


def test_sp500(steps=1000, show_plot=False, verbose=False):
    T = steps
    problem = tigerforecast.problem("SP500-v0")
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    
    if show_plot:
        plt.plot(test_output)
        plt.title("S&P 500")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    print("test_sp500 passed")
    return


if __name__=="__main__":
    test_sp500(show_plot=True)
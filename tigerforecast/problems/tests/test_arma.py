# test the ARMA problem class

import tigerforecast
import jax.numpy as np
import matplotlib.pyplot as plt


def test_arma(steps=1000, show_plot=False, verbose=False):
    T = steps
    p, q = 3, 3
    n = 1
    problem = tigerforecast.problems.ARMA()
    problem.initialize(p,q, n = n)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())
        #print(float(test_output[t]))

    info = problem.hidden()
    if verbose:
        print(info)

    assert problem.T == T
    if verbose:
        print(problem.phi)
        print(problem.psi)
    if show_plot:
        plt.plot(test_output)
        plt.title("arma")
        plt.show(block=False)
        plt.pause(10)
        plt.close()
    print("test_arma passed")
    return


if __name__=="__main__":
    test_arma()
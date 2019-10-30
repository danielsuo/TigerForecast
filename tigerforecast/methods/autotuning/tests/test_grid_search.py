"""
unit tests for GridSearch class
""" 

import tigerforecast
from tigerforecast.methods.autotuning import GridSearch
from tigerforecast.methods.optimizers import *
import jax.numpy as np
import matplotlib.pyplot as plt
import itertools

def test_grid_search(show=False):
    test_grid_search_lstm(show=show)
    test_grid_search_arma(show=show)
    print("test_grid_search passed")

def test_grid_search_lstm(show=False):
    problem_id = "SP500-v0"
    method_id = "LSTM"
    problem_params = {} # {'p':4, 'q':1} # params for ARMA problem
    method_params = {'n':1, 'm':1}
    loss = lambda a, b: np.sum((a-b)**2)
    search_space = {'l': [3, 4, 5, 6], 'h': [2, 5, 8], 'optimizer':[]} # parameters for ARMA method
    opts = [Adam, Adagrad, ONS, OGD]
    lr_start, lr_stop = -1, -3 # search learning rates from 10^start to 10^stop 
    learning_rates = np.logspace(lr_start, lr_stop, 1+2*np.abs(lr_start - lr_stop))
    for opt, lr in itertools.product(opts, learning_rates):
        search_space['optimizer'].append(opt(learning_rate=lr)) # create instance and append

    trials, min_steps = 5, 100
    hpo = GridSearch() # hyperparameter optimizer
    optimal_params, optimal_loss = hpo.search(method_id, method_params, problem_id, problem_params, loss, 
        search_space, trials=trials, smoothing=10, min_steps=min_steps, verbose=show) # run each model at least 1000 steps

    if show:
        print("optimal params: ", optimal_params)
        print("optimal loss: ", optimal_loss)

    # test resulting method params
    method = tigerforecast.method(method_id)
    method.initialize(**optimal_params)
    problem = tigerforecast.problem(problem_id)
    x = problem.initialize(**problem_params)
    loss = []
    if show:
        print("run final test with optimal parameters")
    for t in range(5000):
        y_pred = method.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        method.update(y_true)
        x = y_true

    if show:
        print("plot results")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(10)
        plt.close()


def test_grid_search_arma(show=False):
    problem_id = "ARMA-v0"
    method_id = "AutoRegressor"
    problem_params = {'p':3, 'q':2}
    method_params = {}
    loss = lambda a, b: np.sum((a-b)**2)
    search_space = {'p': [1,2,3,4,5], 'optimizer':[]} # parameters for ARMA method
    opts = [Adam, Adagrad, ONS, OGD]
    lr_start, lr_stop = 0, -4 # search learning rates from 10^start to 10^stop 
    learning_rates = np.logspace(lr_start, lr_stop, 1+2*np.abs(lr_start - lr_stop))
    for opt, lr in itertools.product(opts, learning_rates):
        search_space['optimizer'].append(opt(learning_rate=lr)) # create instance and append

    trials, min_steps = 25, 250
    hpo = GridSearch() # hyperparameter optimizer
    optimal_params, optimal_loss = hpo.search(method_id, method_params, problem_id, problem_params, loss, 
        search_space, trials=trials, smoothing=10, min_steps=min_steps, verbose=show) # run each model at least 1000 steps

    if show:
        print("optimal params: ", optimal_params)
        print("optimal loss: ", optimal_loss)

    # test resulting method params
    method = tigerforecast.method(method_id)
    method.initialize(**optimal_params)
    problem = tigerforecast.problem(problem_id)
    x = problem.initialize(**problem_params)
    loss = []
    if show:
        print("run final test with optimal parameters")
    for t in range(5000):
        y_pred = method.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        method.update(y_true)
        x = y_true

    if show:
        print("plot results")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(10)
        plt.close()



if __name__ == "__main__":
    test_grid_search(show=True)


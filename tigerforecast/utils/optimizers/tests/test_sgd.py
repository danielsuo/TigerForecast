import tigerforecast
from tigerforecast.utils.optimizers.sgd import SGD
from tigerforecast.utils.optimizers.losses import mse
import matplotlib.pyplot as plt

def test_sgd(show=False):
    test_sgd_lstm(show=show)
    test_sgd_autoregressor(show=show)
    print("test_sgd passed")

def test_sgd_lstm(show=False):
    problem = tigerforecast.problem('ARMA-v0')
    x = problem.initialize(p=2,q=0)

    method = tigerforecast.method('LSTM')
    method.initialize(n=1, m=1, l=3, h=10, optimizer=SGD) # initialize with class
    method.predict(1.0) # call methods to verify it works
    method.update(1.0)

    optimizer = SGD(learning_rate=0.001)
    method = tigerforecast.method('LSTM')
    method.initialize(n=1, m=1, l=3, h=10, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = method.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        method.update(y_true)
        x = y_true

    if show:
        plt.title("Test SGD on ARMA(3) with LSTM method")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def test_sgd_autoregressor(show=False):
    problem = tigerforecast.problem('ARMA-v0')
    x = problem.initialize(p=2,q=0)

    optimizer = SGD(learning_rate=0.0003)
    method = tigerforecast.method('AutoRegressor')
    method.initialize(p=3, optimizer=optimizer) # reinitialize with instance

    loss = []
    for t in range(1000):
        y_pred = method.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        method.update(y_true)
        x = y_true

    if show:
        plt.title("Test SGD on ARMA(3) with AutoRegressor method")
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(3)
        plt.close()



if __name__ == "__main__":
    test_sgd(show=True)
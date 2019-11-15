# test the CustomMethod class

import tigerforecast
import jax.numpy as np
import matplotlib.pyplot as plt

# test a simple CustomMethod that returns last value by storing a single param
def test_custom_method(steps=1000, show_plot=True):
    # initial preparation
    T = steps 
    p, q = 3, 3
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
    problem = tigerforecast.problems.ARMA()
    cur_x = problem.initialize(p, q)

    # simple LastValue custom method implementation
    class Custom(tigerforecast.CustomMethod):
        def initialize(self):
            self.x = 0.0
        def predict(self, x):
            self.x = x
            return self.x
        def update(self, y):
            pass

    custom_method = Custom()
    custom_method.initialize()

    # regular LastValue method as sanity check
    reg_method = tigerforecast.methods.LastValue()
    reg_method.initialize()
 
    results = []
    for i in range(T):
        cur_y_pred = custom_method.predict(cur_x)
        reg_y_pred = reg_method.predict(cur_x)
        assert cur_y_pred == reg_y_pred # check that CustomMethod outputs the correct thing
        cur_y_true = problem.step()
        custom_method.update(cur_y_true)
        reg_method.update(cur_y_true)
        results.append(loss(cur_y_true, cur_y_pred))
        cur_x = cur_y_true

    if show_plot:
        plt.plot(results)
        plt.title("Custom (last value) method on ARMA problem")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_custom_method passed")
    return

if __name__=="__main__":
    test_custom_method()


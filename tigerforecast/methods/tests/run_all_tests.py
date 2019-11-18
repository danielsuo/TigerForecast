""" all methods tests """

from tigerforecast.methods.tests.test_last_value import test_last_value
from tigerforecast.methods.tests.test_predict_zero import test_predict_zero
from tigerforecast.methods.tests.test_autoregressor import test_autoregressor
from tigerforecast.methods.tests.test_rnn import test_rnn
from tigerforecast.methods.tests.test_lstm import test_lstm
from tigerforecast.methods.tests.test_custom_method import test_custom_method

def run_all_tests(steps=1000, show=False):
    print("\nrunning all methods tests...\n")
    test_last_value(steps=1000, show_plot=show)
    test_predict_zero(steps=1000, show_plot=show)
    test_autoregressor(steps=1000, show_plot=show)
    #test_rnn(steps=1000, show_plot=show)
    #test_lstm(steps=1000, show_plot=show)
    test_custom_method(steps=steps, show_plot=show)
    print("\nall methods tests passed\n")

if __name__ == "__main__":
    run_all_tests(show=False)

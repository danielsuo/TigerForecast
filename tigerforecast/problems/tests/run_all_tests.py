""" test all problems """
from tigerforecast.problems.tests.test_arma import test_arma
from tigerforecast.problems.tests.test_random import test_random
from tigerforecast.problems.tests.test_sp500 import test_sp500
from tigerforecast.problems.tests.test_uci_indoor import test_uci_indoor
from tigerforecast.problems.tests.test_crypto import test_crypto
from tigerforecast.problems.tests.test_enso import test_enso
from tigerforecast.problems.tests.test_lds_time_series import test_lds_time_series
from tigerforecast.problems.tests.test_lstm_time_series import test_lstm_time_series
from tigerforecast.problems.tests.test_rnn_time_series import test_rnn_time_series
from tigerforecast.problems.tests.test_unemployment import test_unemployment
from tigerforecast.problems.tests.test_custom_problem import test_custom_problem

# run all unit tests for problems
def run_all_tests(steps=1000, show=False):
    print("\nrunning all problems tests...\n")
    test_arma(steps=steps, show_plot=show)
    test_random(steps=steps, show_plot=show)
    test_sp500(steps=steps, show_plot=show)
    test_uci_indoor(steps=steps, show_plot=show)
    test_crypto(steps=steps, show_plot=show)
    test_enso(steps=steps, show_plot=show)
    test_lds_time_series(steps=steps, show_plot=show)
    test_lstm_time_series(steps=steps, show_plot=show)
    test_rnn_time_series(steps=steps, show_plot=show)
    test_unemployment(steps=steps, show_plot=show)
    test_custom_problem(show=show)
    print("\nall problems tests passed\n")
  
if __name__ == "__main__":
    run_all_tests(show=False)


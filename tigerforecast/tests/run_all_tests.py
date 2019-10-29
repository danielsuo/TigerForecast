"""
Run all tests for the TigerForecast framework
"""

from tigerforecast.utils.tests.run_all_tests import run_all_tests as utils_tests
from tigerforecast.problems.tests.run_all_tests import run_all_tests as problems_tests
from tigerforecast.methods.tests.run_all_tests import run_all_tests as methods_tests
from tigerforecast.experiments.tests.run_all_tests import run_all_tests as experiments_tests
from tigerforecast.tests.test_tigerforecast_functionality import test_tigerforecast_functionality

# run all sub-level tests
def run_all_tests(show_results=False):

    print("\n----- Running all TigerForecast tests! -----\n")

    utils_tests(show=show_results)
    experiments_tests(show=show_results)
    methods_tests(show=show_results)
    problems_tests(show=show_results)
    test_tigerforecast_functionality()

    print("\n----- Tests done -----\n")

if __name__ == "__main__":
    run_all_tests(show_results=False)
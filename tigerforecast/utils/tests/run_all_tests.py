import os
import re
from tigerforecast.utils.tests.test_dataset_registry import test_dataset_registry
from tigerforecast.utils.tests.test_download_tools import test_download_tools
from tigerforecast.utils.tests.test_random import test_random
from tigerforecast.utils.tests.test_registration_tools import test_registration_tools
from tigercontrol.utils.optimizers.tests.run_all_tests import run_all_tests as optimizers_tests
from tigercontrol.utils.boosting.tests.run_all_tests import run_all_tests as boosting_tests

# add all unit tests in dataset_registry
def run_all_tests(show=False):
    print("\nrunning all utils tests...\n")
    test_dataset_registry(show=show)
    test_download_tools(show=show)
    test_random(show=show)
    test_registration_tools(show=show)
    optimizers_tests(steps=steps, show=show)
    boosting_tests()
    print("\nall utils tests passed\n")

if __name__ == '__main__':
    run_all_tests(show=False)

# Problem class
# Author: John Hallman

from tigerforecast import error
from tigerforecast.problems import Problem
 
class CustomProblem(object):
    ''' 
    Description: class for implementing algorithms with enforced modularity 
    '''

    def __init__(self):
        pass

def _verify_valid_problem(problem_class):
    ''' 
    Description: verifies that a given class has the necessary minimum problem methods

    Args: a class
    '''
    assert issubclass(problem_class, CustomProblem)
    for f in ['initialize', 'step']:
        if not callable(getattr(problem_class, f, None)):
            raise error.InvalidClass("CustomProblem is missing required method \'{}\'".format(f))



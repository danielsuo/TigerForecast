# Problem class
# Author: John Hallman

from tigerforecast import error
from tigerforecast.problems import Problem

class TimeSeriesProblem(Problem):
    ''' Description: class for online control tests '''
    def initialize(self, **kwargs):
        ''' Description: resets problem to time 0 '''
        self.has_regressors = None
        raise NotImplementedError

    def step(self, action=None):
        ''' Description: Run one timestep of the problem's dynamics. '''
        raise NotImplementedError

    def __str__(self):
    	return "<TimeSeriesProblem>"

    def __repr__(self):
        return self.__str__()
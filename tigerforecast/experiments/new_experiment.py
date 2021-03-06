# NewExperiment class

from tigerforecast import error
from tigerforecast.experiments.core import to_dict, run_experiments, create_full_problem_to_methods

class NewExperiment(object):
    ''' Description: class for implementing algorithms with enforced modularity '''
    def __init__(self):
        self.initialized = False
        
    def initialize(self, problems, methods, problem_to_methods=None, metrics='mse', \
                n_runs = 1, timesteps = None, verbose = 0):
        '''
        Description: Initializes the new experiment instance. 

        Args:     
            problems (dict): map of the form problem_id -> hyperparameters for problem 
            methods (dict): map of the form method_id -> hyperparameters for method
            problem_to_methods (dict) : map of the form problem_id -> list of method_id.
                                       If None, then we assume that the user wants to
                                       test every method in method_to_params against every
                                       problem in problem_to_params
        '''
        self.intialized = True
        self.problems, self.methods, self.metrics = problems, methods, metrics
        self.n_runs, self.timesteps, self.verbose = n_runs, timesteps, verbose

        if(problem_to_methods is None):
            self.problem_to_methods = create_full_problem_to_methods(self.problems.keys(), self.methods.keys())
        else:
            self.problem_to_methods = problem_to_methods

    def run_all_experiments(self):
        '''
        Descripton: Runs all experiments and returns results

        Args:
            None

        Returns:
            prob_method_to_result (dict): Dictionary containing results for all specified metrics and performance
                                         (time and memory usage) for all problem-method associations.
        '''
        prob_method_to_result = {}
        for metric in self.metrics:
            for problem_id in self.problems.keys():
                for (new_problem_id, problem_params) in self.problems[problem_id]:
                    for method_id in self.problem_to_methods[problem_id]:
                        for (new_method_id, method_params) in self.methods[method_id]:
                            loss, time, memory = run_experiments((problem_id, problem_params), (method_id, method_params), \
                                metric, n_runs = self.n_runs, timesteps = self.timesteps, verbose = self.verbose)
                            prob_method_to_result[(metric, problem_id, method_id)] = loss
                            prob_method_to_result[('time', problem_id, method_id)] = time
                            prob_method_to_result[('memory', problem_id, method_id)] = memory

        return prob_method_to_result

    def __str__(self):
        return "<NewExperiment Method>"

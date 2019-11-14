import argparse
import tigerforecast
from tigerforecast.experiments import Experiment
from tigerforecast.utils.optimizers import *

def test_all_methods(problem_name, filename, verbose = 1, lr_tuning = False):
	exp = Experiment()
	exp.initialize(n_runs = 3, verbose = verbose)
	exp.add_problem('MyProblem-v0', {'file' : filename}, name = problem_name)
	exp.add_all_method_variants('AutoRegressor', lr_tuning = lr_tuning)
	exp.add_all_method_variants('LSTM', lr_tuning = lr_tuning)
	exp.scoreboard()
	exp.graph(size = 6)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--problem_name", default=None, type=str, help="problem name")
	parser.add_argument("--filename", default=None, type=str, help="data file to test methods on")
	parser.add_argument("--verbose", default=1, type=str, help="select level of verbosity")
	parser.add_argument("--lr_tuning", default=False, type=str, help="whether or not to tune the learning rate")
	args = vars(parser.parse_args())

	test_all_methods(args["problem_name"], args["filename"], \
		verbose = args["verbose"], lr_tuning = args["lr_tuning"])
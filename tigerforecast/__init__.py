# tigerforecast init file

import os
import sys
import warnings

import tigerforecast
from tigerforecast import error
from tigerforecast.problems import CustomProblem
from tigerforecast.methods import CustomMethod
from tigerforecast.utils.optimizers import losses
from tigerforecast.utils import set_key
from tigerforecast.experiments import Experiment

# initialize global random key by seeding the jax random number generator
# note: numpy is necessary because jax RNG is deterministic
import jax.random as random
GLOBAL_RANDOM_KEY = random.PRNGKey(0)
set_key()


__all__ = [ 
	"CustomMethod", 
	"Experiment", 
	"set_key"
]

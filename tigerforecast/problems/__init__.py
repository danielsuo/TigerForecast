# problems init file

from tigerforecast.problems.core import Problem
from tigerforecast.problems.sp500 import SP500
from tigerforecast.problems.uci_indoor import UCI_Indoor
from tigerforecast.problems.enso import ENSO
from tigerforecast.problems.crypto import Crypto
from tigerforecast.problems.random import Random
from tigerforecast.problems.arma import ARMA
from tigerforecast.problems.unemployment import Unemployment
from tigerforecast.problems.lds_time_series import LDS_TimeSeries
from tigerforecast.problems.rnn_time_series import RNN_TimeSeries
from tigerforecast.problems.lstm_time_series import LSTM_TimeSeries
from tigerforecast.problems.my_problem import MyProblem

# registration tools
from tigerforecast.problems.registration import problem_registry, problem_register, problem
from tigerforecast.problems.custom import register_custom_problem, CustomProblem


# ---------- Time-series ----------


problem_register(
    id='Random-v0',
    entry_point='tigerforecast.problems:Random',
)

problem_register(
    id='MyProblem-v0',
    entry_point='tigerforecast.problems:MyProblem',
)

problem_register(
    id='ARMA-v0',
    entry_point='tigerforecast.problems:ARMA',
)

problem_register(
    id='SP500-v0',
    entry_point='tigerforecast.problems:SP500',
)

problem_register(
    id='UCI-Indoor-v0',
    entry_point='tigerforecast.problems:UCI_Indoor',
)

problem_register(
    id='Crypto-v0',
    entry_point='tigerforecast.problems:Crypto',
)

problem_register(
    id='Unemployment-v0',
    entry_point='tigerforecast.problems:Unemployment',
)

problem_register(
    id='ENSO-v0',
    entry_point='tigerforecast.problems:ENSO',
)

problem_register(
    id='LDS-TimeSeries-v0',
    entry_point='tigerforecast.problems:LDS_TimeSeries',
)

problem_register(
    id='RNN-TimeSeries-v0',
    entry_point='tigerforecast.problems:RNN_TimeSeries',
)

problem_register(
    id='LSTM-TimeSeries-v0',
    entry_point='tigerforecast.problems:LSTM_TimeSeries',
)




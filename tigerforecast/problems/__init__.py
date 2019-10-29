# problems init file

from tigerforecast.problems.core import Problem
from tigerforecast.problems.registration import problem_registry, problem_register, problem
from tigerforecast.problems.time_series import TimeSeriesProblem
from tigerforecast.problems.custom import register_custom_problem, CustomProblem


# ---------- Time-series ----------


problem_register(
    id='Random-v0',
    entry_point='tigerforecast.problems.time_series:Random',
)

problem_register(
    id='MyProblem-v0',
    entry_point='tigerforecast.problems.time_series:MyProblem',
)

problem_register(
    id='ARMA-v0',
    entry_point='tigerforecast.problems.time_series:ARMA',
)

problem_register(
    id='SP500-v0',
    entry_point='tigerforecast.problems.time_series:SP500',
)

problem_register(
    id='UCI-Indoor-v0',
    entry_point='tigerforecast.problems.time_series:UCI_Indoor',
)

problem_register(
    id='Crypto-v0',
    entry_point='tigerforecast.problems.time_series:Crypto',
)

problem_register(
    id='Unemployment-v0',
    entry_point='tigerforecast.problems.time_series:Unemployment',
)

problem_register(
    id='ENSO-v0',
    entry_point='tigerforecast.problems.time_series:ENSO',
)

problem_register(
    id='LDS-TimeSeries-v0',
    entry_point='tigerforecast.problems.time_series:LDS_TimeSeries',
)

problem_register(
    id='RNN-TimeSeries-v0',
    entry_point='tigerforecast.problems.time_series:RNN_TimeSeries',
)

problem_register(
    id='LSTM-TimeSeries-v0',
    entry_point='tigerforecast.problems.time_series:LSTM_TimeSeries',
)





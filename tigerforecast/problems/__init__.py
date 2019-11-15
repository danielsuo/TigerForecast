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
from tigerforecast.problems.custom import CustomProblem
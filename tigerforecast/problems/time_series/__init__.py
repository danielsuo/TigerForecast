# time_series init file

from tigerforecast.problems.time_series.time_series_problem import TimeSeriesProblem
from tigerforecast.problems.time_series.sp500 import SP500
from tigerforecast.problems.time_series.uci_indoor import UCI_Indoor
from tigerforecast.problems.time_series.enso import ENSO
from tigerforecast.problems.time_series.crypto import Crypto
from tigerforecast.problems.time_series.random import Random
from tigerforecast.problems.time_series.arma import ARMA
from tigerforecast.problems.time_series.unemployment import Unemployment
from tigerforecast.problems.time_series.lds_time_series import LDS_TimeSeries
from tigerforecast.problems.time_series.rnn_time_series import RNN_TimeSeries
from tigerforecast.problems.time_series.lstm_time_series import LSTM_TimeSeries
from tigerforecast.problems.time_series.my_problem import MyProblem
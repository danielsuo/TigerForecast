# time_series init file

from tigerforecast.methods.core import Method
from tigerforecast.methods.autoregressor import AutoRegressor
from tigerforecast.methods.least_squares import LeastSquares
from tigerforecast.methods.last_value import LastValue
from tigerforecast.methods.predict_zero import PredictZero
from tigerforecast.methods.rnn import RNN
from tigerforecast.methods.lstm import LSTM
from tigerforecast.methods.wave_filtering import WaveFiltering
from tigerforecast.methods.boosting.simple_boost import SimpleBoost
from tigerforecast.methods.prophet import Prophet
from tigerforecast.methods.custom import CustomMethod

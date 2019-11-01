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

# registration tools
from tigerforecast.methods.registration import method_registry, method_register, method
from tigerforecast.methods.custom import CustomMethod, register_custom_method


# ---------- Time-Series Methods ----------

method_register(
    id='WaveFiltering',
    entry_point='tigerforecast.methods:WaveFiltering',
)

method_register(
    id='LastValue',
    entry_point='tigerforecast.methods:LastValue',
)

method_register(
    id='LeastSquares',
    entry_point='tigerforecast.methods:LeastSquares',
)

method_register(
    id='AutoRegressor',
    entry_point='tigerforecast.methods:AutoRegressor',
)

method_register(
    id='PredictZero',
    entry_point='tigerforecast.methods:PredictZero',
)

method_register(
    id='RNN',
    entry_point='tigerforecast.methods:RNN',
)

method_register(
    id='LSTM',
    entry_point='tigerforecast.methods:LSTM',
)


# ---------- Boosting Methods ----------


method_register(
    id='SimpleBoost',
    entry_point='tigerforecast.methods.boosting:SimpleBoost',
)

method_register(
    id='SimpleBoostAdj',
    entry_point='tigerforecast.methods.boosting:SimpleBoostAdj',
)

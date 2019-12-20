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
from tigerforecast.methods.seq2seqlstmstateless import Seq2seqLSTMStateless
from tigerforecast.methods.floodlstm_Alex import FloodLSTM
from tigerforecast.methods.ARStateless import ARStateless
from tigerforecast.methods.FloodAR import FloodAR
#from tigerforecast.methods.prophet import Prophet

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

method_register(
    id='Seq2seqLSTMStateless',
    entry_point='tigerforecast.methods:Seq2seqLSTMStateless',
)

method_register(
    id='FloodLSTM',
    entry_point='tigerforecast.methods:FloodLSTM',
)

method_register(
    id='ARStateless',
    entry_point='tigerforecast.methods:ARStateless',
)

method_register(
    id='FloodAR',
    entry_point='tigerforecast.methods:FloodAR',
)

method_register(
    id='Prophet',
    entry_point='tigerforecast.methods:Prophet',
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

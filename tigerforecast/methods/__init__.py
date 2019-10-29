# methods init file

from tigerforecast.methods.registration import method_registry, method_register, method
from tigerforecast.methods.core import Method
from tigerforecast.methods.custom import CustomMethod, register_custom_method
from tigerforecast.methods.optimizers import losses


# ---------- Time-Series Methods ----------

method_register(
    id='WaveFiltering',
    entry_point='tigerforecast.methods.time_series:WaveFiltering',
)

method_register(
    id='LastValue',
    entry_point='tigerforecast.methods.time_series:LastValue',
)

method_register(
    id='LeastSquares',
    entry_point='tigerforecast.methods.time_series:LeastSquares',
)

method_register(
    id='AutoRegressor',
    entry_point='tigerforecast.methods.time_series:AutoRegressor',
)

method_register(
    id='PredictZero',
    entry_point='tigerforecast.methods.time_series:PredictZero',
)

method_register(
    id='RNN',
    entry_point='tigerforecast.methods.time_series:RNN',
)

method_register(
    id='LSTM',
    entry_point='tigerforecast.methods.time_series:LSTM',
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


# ---------- Control Methods ----------


method_register(
    id='KalmanFilter',
    entry_point='tigerforecast.methods.control:KalmanFilter',
)

method_register(
    id='ODEShootingMethod',
    entry_point='tigerforecast.methods.control:ODEShootingMethod',
)

method_register(
    id='LQR',
    entry_point='tigerforecast.methods.control:LQR',
)

method_register(
    id='ILQR',
    entry_point='tigerforecast.methods.control:ILQR',
)

method_register(
    id='CartPoleNN',
    entry_point='tigerforecast.methods.control:CartPoleNN',
)

method_register(
    id='GPC',
    entry_point='tigerforecast.methods.control:GPC',
)


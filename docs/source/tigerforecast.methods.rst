tigerforecast.methods package
===========================

.. automodule:: tigerforecast.methods

core
----
.. autosummary::
  :toctree: _autosummary

   Method

control
-------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.methods.control.ControlMethod
   tigerforecast.methods.control.KalmanFilter
   tigerforecast.methods.control.ODEShootingMethod
   tigerforecast.methods.control.LQR
   tigerforecast.methods.control.MPPI
   tigerforecast.methods.control.CartPoleNN
   tigerforecast.methods.control.ILQR


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.methods.time_series.TimeSeriesMethod
   tigerforecast.methods.time_series.AutoRegressor
   tigerforecast.methods.time_series.LastValue
   tigerforecast.methods.time_series.PredictZero
   tigerforecast.methods.time_series.RNN
   tigerforecast.methods.time_series.LSTM
   tigerforecast.methods.time_series.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.methods.optimizers.Optimizer
   tigerforecast.methods.optimizers.Adagrad
   tigerforecast.methods.optimizers.Adam
   tigerforecast.methods.optimizers.ONS
   tigerforecast.methods.optimizers.SGD
   tigerforecast.methods.optimizers.OGD
   tigerforecast.methods.optimizers.mse
   tigerforecast.methods.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  tigerforecast.methods.boosting.SimpleBoost

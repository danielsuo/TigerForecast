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

   tigerforecast.methods.Method
   tigerforecast.methods.AutoRegressor
   tigerforecast.methods.LastValue
   tigerforecast.methods.PredictZero
   tigerforecast.methods.RNN
   tigerforecast.methods.LSTM
   tigerforecast.methods.LeastSquares

optimizers
----------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.utils.optimizers.Optimizer
   tigerforecast.utils.optimizers.Adagrad
   tigerforecast.utils.optimizers.Adam
   tigerforecast.utils.optimizers.ONS
   tigerforecast.utils.optimizers.SGD
   tigerforecast.utils.optimizers.OGD
   tigerforecast.utils.optimizers.mse
   tigerforecast.utils.optimizers.cross_entropy

boosting
--------

.. autosummary::
  :toctree: _autosummary

  tigerforecast.utils.boosting.SimpleBoost

tigerforecast.problems package
=============================

.. automodule:: tigerforecast.problems

core
----

This is a core

.. autosummary::
  :toctree: _autosummary

   Problem

custom
------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.problems.CustomProblem
   tigerforecast.problems.register_custom_problem


control
-------

.. toctree::
   :maxdepth: 3

.. autosummary::
  :toctree: _autosummary

   tigerforecast.problems.ControlProblem
   tigerforecast.problems.control.LDS_Control
   tigerforecast.problems.control.LSTM_Control
   tigerforecast.problems.control.RNN_Control
   tigerforecast.problems.control.CartPole
   tigerforecast.problems.control.DoublePendulum
   tigerforecast.problems.control.Pendulum


time_series
-----------

.. autosummary::
  :toctree: _autosummary

   tigerforecast.problems.TimeSeriesProblem
   tigerforecast.problems.time_series.SP500
   tigerforecast.problems.time_series.UCI_Indoor
   tigerforecast.problems.time_series.ENSO
   tigerforecast.problems.time_series.Crypto
   tigerforecast.problems.time_series.Random
   tigerforecast.problems.time_series.ARMA
   tigerforecast.problems.time_series.Unemployment
   tigerforecast.problems.time_series.LDS_TimeSeries
   tigerforecast.problems.time_series.LSTM_TimeSeries
   tigerforecast.problems.time_series.RNN_TimeSeries

pybullet
--------

.. autosummary::
  :toctree: _autosummary

  tigerforecast.problems.pybullet.PyBulletProblem
  tigerforecast.problems.pybullet.Simulator
  tigerforecast.problems.pybullet.Ant
  tigerforecast.problems.pybullet.CartPole
  tigerforecast.problems.pybullet.CartPoleDouble
  tigerforecast.problems.pybullet.CartPoleSwingup
  tigerforecast.problems.pybullet.HalfCheetah
  tigerforecast.problems.pybullet.Humanoid
  tigerforecast.problems.pybullet.Kuka
  tigerforecast.problems.pybullet.KukaDiverse
  tigerforecast.problems.pybullet.Minitaur
  tigerforecast.problems.pybullet.Obstacles
  


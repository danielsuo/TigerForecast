# TigerForecast
**********

**TigerForecast** is an open-source framework for benchmarking time-series algorithms in simulated and real settings, and is available for anyone to download and use. By reducing algorithms to a set of standard APIs, TigerForecast allows the user to quickly switch between methods and tasks while running experiments and plotting results on the go, and for quick and simple comparison between method performances. TigerForecast also comes with built-in standard time-series algorithms for comparison or other use.
The *main algorithmic innovation* is boosting: TigerForcast allows you to take any forcasting method (including third party) and improve its accuracy using the techniques developed in https://arxiv.org/abs/1906.08720. 


Overview
========

Although there are several machine learning platforms that aid with the implementation of algorithms, there are far fewer readily available tools for benchmarks and comparisons. The main implementation frameworks (eg. Keras, PyTorch) provide certain small-scale tests, but these are in general heavily biased towards the batch setting. TigerForecast exists to fill this gap — we provide a variety of evaluation settings to test online time-series algorithms on, with more diversity and generality than any other online benchmarking platform.


Installation
============

Clone the directory and use pip or pip3 to set up a minimal installation of TigerForecast, which excludes PyBullet control problems.

```
    git clone https://github.com/MinRegret/tigerforecast.git
    pip install -e tigerforecast
```

You can now use TigerForecast in your Python code by calling `import tigerforecast` in your files. 

Finally, run a demo to verify that the installation was successful!

```
    python tigerforecast/problems/tests/test_arma.py
```


For more information
====================

To learn more about TigerForecast and how to incorporate it into your research, check out the Quickstart guide in the ```tigerforecast/tutorials``` folder. Alternatively, check out our [readthedocs](https://tigerforecast.readthedocs.io/en/latest/) page for more documentation and APIs.


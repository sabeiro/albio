---
title: "Albio library description"
author: Giovanni Marelli
date: 2019-01-12
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# Albio

![lernia logo](docs/f_albio/logo_albio.svg "albio logo")

Albio is a time series analysis library for statistical properties calculation and forecasts 

![module_lernia](docs/f_albio/module_albio.svg "albio_module")
albio module_


## modules descriptions

> time series

* `series_load.py`
	* load and preprocess time series mainly from web services
* `series_stat.py`
	* statistical properties and filtering of time series
* `series_forecast.py`
	* forecast on time series (arima, holt-winter, bayesian...)
* `series_neural.py`
	* forecast based on neural networks
* `algo_holtwinters.py`
	* implementation of holt-winters algorithm


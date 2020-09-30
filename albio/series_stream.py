import numpy as np
import pandas as pd
import datetime
from scipy.optimize import leastsq as least_squares
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

class stream:
    """buffering stream data"""
    def __init__(self,n_buf=5):
        """set the buffer"""
        self.buf = np.array(range(n_buf))

    def set(self,y):
        """add a new value"""
        if y == float('nan'): return
        buf = np.roll(self.buf,-1)
        buf[-1] = y
        self.buf = buf

    def get(self):
        """get current average"""
        return np.mean(self.buf)

    def new(self,y):
        """new average after insertion"""
        self.set(y)
        return self.get(y)

def streamList(col,n_buf):
    """stream a list - spark"""
    buf = [0. for x in range(n_buf)]
    norm = 1./n_buf
    for c in col:
        buf = buf[1:] + [c]
        c = sum(buf)*norm

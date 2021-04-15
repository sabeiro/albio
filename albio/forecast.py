import numpy as np
import pandas as pd
import datetime
from scipy.optimize import leastsq as least_squares
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

def standardConf():
    """standard configuration dictionary for the class"""
    conf = {"arima":(1,2,1)}

def to_date(t,dt="seconds"):
    """convert list into datetime"""
    today = datetime.datetime(2020, 1, 1)
    ts = [today + datetime.timedelta(days=i+1) for i in t]
    return ts
    
def add_delta(t,n=14,dt="seconds"):
    """add n days to existing prediction"""
    if dt == "days":
        ts = [list(t)[-1] + datetime.timedelta(days=i+1) for i in range(n)]
    elif dt == "seconds":
        ts = [list(t)[-1] + datetime.timedelta(seconds=i+1) for i in range(n)]
    ts = list(t) + ts
    dt_idx = pd.DatetimeIndex(ts)
    return dt_idx
    
    
class forecast():
    """iterates and compares different forecast methods"""
    def __init__(self, t, X, y):
        self.t = t
        self.X = X
        self.y = y

    def lstm(self,*args,n=8):
        """long short time memory"""
        try:
            from keras.models import Sequential
            from keras.layers import LSTM,Dense
            from keras.layers import Dropout
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
        except:
            print('tensorflow/keras not installed')
            return t, y, y
        dataset = pd.DataFrame(self.y)
        data = np.array(self.y).reshape(-1, 1)
        train_data = dataset[:len(dataset)-n]
        test_data = dataset[len(dataset)-n:]
        scaler = MinMaxScaler()
        scaler.fit(dataset)
        self.scaler = scaler
        scaled_train_data = scaler.transform(train_data)
        n_input, n_features = n, 1
        generator = TimeseriesGenerator(scaled_train_data,scaled_train_data,length=n,batch_size=1)
        model = Sequential()
        model.add(LSTM(units = 32, return_sequences = True, input_shape = (n_input, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 32, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 32))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(generator, epochs = 21)
        losses_lstm = model.history.history['loss']
        self.lstm_model = model

    def lstm_forecast(self,t,X):
        """forecast on lmts"""
        n_input, n_features = len(t), X.shape[1]
        lstm_predictions_scaled = []
        scaled_test_data  = self.scaler.transform(X)
        batch = scaled_test_data
        current_batch = batch.reshape((1, n_input, n_features))
        for i in range(n_input):
            print(current_batch)
            lstm_pred = self.lstm_model.predict(current_batch)[0]
            lstm_predictions_scaled.append(lstm_pred) 
            current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
        y_pred = pd.DataFrame(self.scaler.inverse_transform(lstm_predictions_scaled))
        return y_pred

    def mlp_regressor(self,*args):
        """mlp regressor on time series"""
        model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
        _ = model.fit(self.X.values, self.y)
        self.mlp_model = model

        
    def mlp_forecast(self,t,X):
        """forecast with multilayer perceptron"""
        y_pred = self.mlp_model.predict(X)
        return y_pred
        

    def prophet(self,dt="seconds",*args):
        """prophet forecast"""
        try:
            from fbprophet import Prophet
            from fbprophet.plot import plot_plotly, add_changepoints_to_plot
        except:
            print("fbprophet not installed")
            return {}
        model = Prophet()
        pr_data = pd.DataFrame({'ds':to_date(self.t,dt=dt),'y':self.y})
        # for i in self.X.columns:
        #     pr_data.loc[:,i] = self.X[i].values
        #     model.add_regressor(i)
        model.fit(pr_data)
        self.pro_model = model

    def prophet_forecast(self,t,X,dt="seconds"):
        """forecast a trained prophet"""
        ts = to_date(t,dt=dt)
        future = pd.DataFrame({"ds":ts})
        # for i in X.columns:
        #     future.loc[:,i] = X[i].values
        # fL = future.columns
        # fore = self.pro_model.predict(future.head(1))
        # foreL = [fore]
        # for i in range(len(t)-1):
        #     print(fore[fL])
        #     fore1 = self.pro_model.predict(fore[fL])
        #     foreL.append(fore1)
        #     fore = fore1
        # forecast = pd.concat(foreL)e
        forecast = self.pro_model.predict(future)
        return forecast

    def arima(self,conf={},*args):
        """arima autoregressive moving average"""
        print(self.y)
        if conf == {}:
            conf['arima'] = (1,2,1)
        try:
            model = ARIMA(self.y, order=conf['arima'])
            arm_model = model.fit(trend='c', full_output=True, disp=True)
        except:
            model = ARIMA(self.y, order=(1, 1, 1))
            arm_model = model.fit(trend='c', full_output=True, disp=True)
        self.arm_model = arm_model
            
    def arima_forecast(self,t,X):
        """forecast using arima"""
        n = len(t)
        forcast = self.arm_model.forecast(steps=n)
        y_pred = forcast[0].tolist()
        return y_pred

##----------------------------trend-functions------------------------------
def ser_poly(t, p):
    """4th order poynom"""
    return p[0] + p[1] * t + p[2] * t ** 2 + p[3] * t ** 3 # + p[4] * t ** 4  # + p[5]*t**5


def ser_residuals(p, t, y):
    """residual function"""
    return (y - ser_poly(t, p))


def ser_sin(t, p, f):  
    """custom sinus function"""
    return p[0] + p[1] * np.sin(f[0] * t + p[2])

def ser_sin2(t, p, f):  # print(2.*np.pi/(7.))
    """custom sinus function"""
    return p[0] + p[1] * np.sin(f[0] * t + p[2]) * (1 + p[3] * np.sin(f[1] * t + p[4]))


def ser_sin_min(p, t, y, f):
    """residual on sinus function"""
    return ser_sin(y, p, f) - y


def ser_exp(t, decay):
    """exponential decay"""
    return np.exp(-decay * t)


def poly_4(t,x,y,n=14,x0=[None],dt="seconds",*args):
    """polynomial 4th grade"""
    if x0[0] == None:
        x0 = np.ones(10)
    t1 = add_delta(t,n=n,dt=dt)
    ts = np.array([x.timestamp() for x in t])
    ts1 = np.array([x.timestamp() for x in t1])
    res_lsq = least_squares(ser_residuals,x0,args=(ts,y))
    poly = [ser_poly(x,res_lsq[0]) for x in ts1]
    return ts, poly, res_lsq[0]

def bi_week(t,x,y,n=14,x0=[None],dt="seconds",*args):
    """bi weekly frequencies"""
    if x0[0] == None:
        x0 = np.ones(10)
    t1 = add_delta(t,n=n,dt=dt)
    ts = [x.timestamp() for x in t]
    ts1 = [x.timestamp() for x in t1]
    f = [(2.*np.pi)/7.,(2.*np.pi)/14.]
    res_lsq = least_squares(ser_sin_min,x0,args=(ts,y,f))
    poly = [ser_sin(x,res_lsq[0],f) for x in ts1]
    return ts, poly, res_lsq[0]


def trendFluct(t,x,y,n=14,conf={},isPlot=False,dt="seconds",*args):
    """predict the signal decomposing the trend from the fluctuations"""
    if conf == {}:
        conf['poly'] = np.ones(10)
        conf['sin'] = np.ones(10)
    orig = t[0].timestamp()
    ts = np.array([x.timestamp() - orig for x in t])
    t1 = add_delta(t,n=n,dt=dt)
    ts1 = np.array([x.timestamp() - orig for x in t1])
    ts = ts/(3600*24)
    ts1 = ts1/(3600*24)
    res_pol = least_squares(ser_residuals,conf['poly'],args=(ts,y))
    poly = np.array([ser_poly(x,res_pol[0]) for x in ts])
    y1 = np.array(y) - poly
    f0 = [(2.*np.pi)/7.,(2.*np.pi)/14.]
    #f = [10.38,5.19]
    f = [f0[0]]
    conf['sin'] = [np.mean(y1),np.max(y1),0.]
    res_sin1 = least_squares(ser_sin_min,conf['sin'],args=(ts,y1,f))
    siny1 = np.array([ser_sin(x,res_sin1[0],f) for x in ts])
    y2 = y1 - siny1
    f = [f0[1]]
    conf['sin'] = [np.mean(y2),np.max(y2),0.]
    res_sin = least_squares(ser_sin_min,conf['sin'],args=(ts,y2,f))
    f = [f0[0]]
    siny = np.array([ser_sin(x,res_sin[0],f) for x in ts1])
    #siny = np.array([ser_sin(x,[165.28191453742843, 16992.993457450488, 0.0],f) for x in ts1])
    f = [f0[1]]
    siny1 = np.array([ser_sin(x,res_sin1[0],f) for x in ts1])
    poly = np.array([ser_poly(x,res_pol[0]) for x in ts1])
    pred = poly + siny + siny1
    conf['poly'] = res_pol[0]
    conf['sin'] = res_sin[0]
    if isPlot:
        plt.plot(t,y,label="series")
        plt.plot(t1,poly,label="polynomial")
        plt.plot(t,y1,label="difference")
        plt.plot(t1,siny,label="1week")
        plt.plot(t1,siny1,label="2week")
        plt.plot(t1,pred,label="prediction")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()
    return t1, pred, conf


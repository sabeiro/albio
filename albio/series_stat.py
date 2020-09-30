"""
series_stat:
filtering and extraction of statistical attributes from time series
"""

import itertools
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from matplotlib import mlab
from scipy.optimize import leastsq as least_squares
from sklearn import linear_model
from sklearn.decomposition import FastICA, PCA

# import series_interp as s_i

def serRunAv(y, steps=5):
    """perform a running average"""
    Nr = int(steps)
    Ntot = len(y)
    y_av = np.convolve(y, np.ones(Nr) / steps, 'same')
    for i in range(0, int(steps / 2)):
        y_av[i] = np.nansum(y[i:(Nr + i)]) / steps
        y_av[Ntot - i - 1] = np.nansum(y[(Ntot - i - Nr):(Ntot - i)]) / steps
    return y_av


def serRunAvDev(y, nInt=5):
    """perform a running average and standard deviation"""
    N = len(y)
    x = np.linspace(0, 1, N)
    xf, yd, yf = [], [], []
    for i in range(nInt):
        dn = int(i * N / nInt)
        dn1 = int((i + 1) * N / nInt)
        yd.append(np.std(y[dn:dn1]))
        xf.append(np.mean(x[dn:dn1]))
        yf.append(np.mean(y[dn:dn1]))
    return x, yd, xf, yf

def serDecompose(y,period=7,isPlot=False):
    """decompose signal"""
    serD = pd.DataFrame({'y':y})
    serD.loc[:,"run_av"] = serRunAv(y,steps=period)
    serD.loc[:,"smooth"] = serSmooth(y,width=3,steps=period)
    serD['diff'] = serD['smooth'] - serD['smooth'].shift()
    serD['grad'] = serD['diff'] - serD['diff'].shift()
    serD['e_av'] = serD['y'].ewm(halflife=period).mean()
    serD['stat'] = serD['run_av'] - serD['diff']
    if isPlot:
        plt.plot(serD['y'],label="raw")
        plt.plot(serD['run_av'],label="run_av")
        plt.plot(serD['smooth'],label="smooth")
        plt.plot(serD['diff']+np.mean(serD['y']),label="diff")
        plt.plot(serD['e_av'],label="e_av")
        plt.plot(serD['stat'],label="stat")
        plt.legend()
        plt.show()
    return serD


def rem_nan(y):
    """remove nans"""
    return [x for x in y if x == x]


def binAvStd(df, col_y="act", col_bin="time"):
    """perform a binned average and standard deviation"""

    def clampF(x):
        return pd.Series({"y": np.mean(x[col_y]), "sy": np.std(x[col_y])})

    dg = df.groupby(col_bin).apply(clampF).reset_index()
    return dg


def serSmooth(y, width=3, steps=5):
    """ kaiser window smoothing """
    N = len(y)
    if N < 3:
        return y
    Nr = max(int(steps), 3)
    Nr2 = max(int(steps / 2), 1)
    s = np.r_[y[Nr - 1:0:-1], y, y[-1:-Nr:-1]]
    w = np.kaiser(Nr, width)
    y1 = np.convolve(w / w.sum(), s, mode='valid')
    Nend = y1.shape[0] - Nr2 if y1.shape[0] - 2 * Nr2 == N else y1.shape[0] - Nr2 + 1
    if False:
        plt.plot(range(N), y, label="y")
        plt.plot(range(N), y1[Nr2:Nend], label="smooth")
        plt.legend()
        plt.show()
        plt.plot(w)
        plt.show()
    return y1[Nr2:Nend]


def serKalman(y, R=0.01 ** 2):
    """Kalman filter"""
    sz = (y.shape[0],)  # size of array
    z = np.array(y)  # observations (normal about x, sigma=0.1)
    Q = 1e-5  # process variance
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor
    # intial guesses
    xhat[0] = y[0]
    P[0] = 1.0
    for k in range(1, sz[0]):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


def addNoise(y, noise=.2):
    """add a portion of random gaussian noise"""
    return y * (1. + np.random.randn(len(y)) * noise)


def phaseLag(y1, y2, isPlot=False, ax=None):
    """returns phase lag between two signals"""
    t = np.linspace(0., 1., len(y1))
    # from dot product
    opp = np.sum(y2 * y1)
    hyp = np.sqrt(np.sum(y2 ** 2) * np.sum(y1 ** 2))
    # from cross product
    adj = np.cross([np.cos(t), y1], [np.cos(t), y2], axis=0)
    phase_angle = np.arctan2(adj, opp)
    n = len(y1)
    t = np.linspace(-0.5*n, 0.5*n, n)
    offset = t[np.argmax(phase_angle)]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        # ax.plot(y1, label="first signal")
        # ax.plot(y2, label="second signal")
        # ax.plot(adj, label="adjust")
        ax.plot(t,phase_angle, label="phase")
        ax.axvline(x=offset,linestyle="-.",color="green")
        ax.axvline(x=0,linestyle="--",color="red")
        ax.set_ylabel('phase lag')
        #ax.legend()
    return offset, phase_angle

def timeLag(y1, y2, isPlot=False, samples_per_second=10, ax=None):
    """time lag between two signals"""
    #w, h = sp.signal.freqs(y1, y2)
    # w, h = sp.signal.freqz(y1, y2, 100)
    # group_delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
    # H = np.abs(h)**2
    a = y1[1:] - y1[:-1]
    b = y2[1:] - y2[:-1]
    a = y1 - np.mean(y1)
    b = y2 - np.mean(y2)
    p_inst = a * b
    s = np.sign(p_inst)
    #s = s[s!=0]
    sa = np.sign(a)
    start = np.sign(np.mean(sa))
    num_samples = len(s)
    freq_Hz = 10.0
    t = np.linspace(0.0, ((num_samples - 1) / samples_per_second), num_samples)
    s_avg = np.trapz(s, x=t) / (t[-1] - t[0])
    shift_fraction = s_avg / 2 + 0.5
    phase_shift = start*(1 - shift_fraction) * np.pi
    offset = t[np.argmax(phase_shift)]
    #phase_shift = shift_fraction * np.pi
    # ab_corr = np.correlate(y1, y2, "full")
    # dt = np.linspace(-t[-1], t[-1], (2 * num_samples) - 1)
    # t_shift_alt = (1.0 / samples_per_second) * ab_corr.argmax() - t[-1]
    # t_shift = dt[ab_corr.argmax()]
    # phase_shift = ((2.0 * np.pi) * ((t_shift / (1.0 / freq_Hz)) % 1.0)) - np.pi
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(p_inst,label="cross correlation")
        ax.axvline(x=offset,linestyle="-.",color="green")
        ax.axvline(x=len(phase_shit)/2,linestyle="--",color="red")
    return offset, phase_shift
    
def maxLag(y1, y2, t=[None], isPlot=False, ax = None):
    """max lag between two signals"""
    n = len(y1)
    norm = np.sqrt(sp.signal.correlate(y1, y1, mode='same')[int(n/2)]
                   * sp.signal.correlate(y2, y2, mode='same')[int(n/2)])
    corr = sp.signal.correlate(y2, y1, mode='same') / norm
    if not any(t):
        t = np.linspace(-0.5*n, 0.5*n, n)
    delay = t[np.argmax(np.abs(corr))]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(t, corr)
        ax.axvline(x=0.,linestyle="--",color="red")
        ax.axvline(x=delay,linestyle="-.",color="green")
        #ax.set_title('Lag: ' + str(np.round(delay, 3)) + ' ')
        ax.set_ylabel('correlation coeff')
    return delay, corr

def groupDelay(y1, y2, isPlot=False, ax=None):
    """group delay between signals"""
    w, group_delay = sp.signal.group_delay((y2, y1))
    n2 = int(len(w)/2)
    w = w - w[n2]
    delay = w[np.argmax(group_delay)]
    if False:
        w, h = sp.signal.freqz(y2, y1, 50)
        group_delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
    if isPlot :
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(w, group_delay)
        ax.axvline(x=0,linestyle="--",color="red")
        ax.axvline(x=delay,linestyle="-.",color="green")
        ax.set_ylabel('group delay')
    return delay, group_delay

def xcorLag(y1, y2, period=64, isPlot=False, ax=None):
    """find the maximum of a cross correlation with lag"""
    #xcor = [xCor( y1,shift(y2,lag) ) for lag in range(period)]
    xcor = [sp.stats.pearsonr( y1,shift(y2,lag) )[0] for lag in range(period)]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(xcor)
        ax.axvline(x=len(xcor)/2,linestyle="--",color="red")
        ax.set_ylabel("cross correlation")
    return xcor


def xcorMax(y1, y2, period=64, isPlot=False, ax=None):
    """find the maximum of a cross correlation with lag"""
    #xcor = [xCor( y1,shift(y2,lag) ) for lag in range(period)]
    t = list(range(-period,period))
    xcor = [sp.stats.pearsonr( y2,shift(y1,lag) )[0] for lag in t]
    n = len(xcor)
    t = np.linspace(-0.5*n, 0.5*n, n)
    offset = t[np.argmax(xcor)]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(t,xcor)
        ax.axvline(x=0,linestyle="--",color="red")
        ax.axvline(x=offset,linestyle="-.",color="green")
        ax.set_ylabel("max cross corr")
        #plt.axvline(0,color='k',linestyle='--',label='Center')
        #plt.axvline(offset,color='r',linestyle='--',label='Peak synchrony')
    return offset, xcor


def xcorFft(y1, y2, isPlot=False, ax=None):
    """time shift using fft"""
    y1 = np.array(y1)
    y2 = np.array(y2)
    af = sp.fft.fft(y1)
    bf = sp.fft.fft(y2)
    xcor = sp.fft.ifft(af * np.conj(bf))
    n = len(xcor)
    t = np.linspace(-0.5*n, 0.5*n, n)
    offset = t[ np.argmax(abs(xcor))]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(t,xcor)
        ax.axvline(x=0,linestyle="--",color="red")
        ax.axvline(x=offset,linestyle="-.",color="green")
        ax.set_ylabel("fft correlation")
    return offset, xcor

def xcorSer(y1, y2, isPlot=False, lab="cross correlation"):
    """peak of cross correlation"""
    from scipy.signal import correlate
    xcorr = correlate(y1, y2)
    dt = np.arange(1-len(y1), len(y1))
    offset = dt[xcorr.argmax()]
    if isPlot:
        if ax == None:
            fig, ax = plt.subplots(1,1)
        ax.plot(dt,xcor)
        ax.axvline(x=len(xcor)/2,linestyle="--",color="red")
        ax.set_ylabel("fft correlation")
    return offset, xcorr

def xcorDecay(y1, y2, period=64, isPlot=False, lab="cross correlation"):
    """signal decay on cross correlation"""
    xcor = xcorLag(y1,y2)
    t = np.linspace(0,len(xcor),len(xcor))
    def ser_exp(x, t, p):
        exp1 = np.exp(-x[0]*t)
        return x[1]*exp1

    def ser_residual(x, t, y, p):
        return (y - ser_exp(x, t, p))

    x0 = [1.,1.]
    p0 = [0.,0.]
    res, _ = least_squares(ser_residual, x0, args=(t, xcor, p0))
    if isPlot:
        plt.plot(xcor,label=lab)
        plt.plot(ser_exp(res,t,p0),label=lab + " interpolation")
        plt.legend()
    return res, ser_exp(res,t,p0)


def delayM(M,period=64):
    """compute cross correlation between matrix columns"""
    #cM = cross_funcM(M, xCor)
    def func(y1,y2):
        #offset, xcor = xcorMax(y1,y2,period=period)
        #offset, xcor = xcorFft(y1,y2)
        offset, xcor = maxLag(y1,y2)
        #offset, xcor = xcorSer(y1,y2)
        return offset
    
    colL = [x for x in M.columns]
    N = len(colL)
    cM = np.zeros((N, N))
    for i in range(N):
        cM[i, i] = func(M[colL[i]], M[colL[i]])
        for j in range(i + 1, N):
            cM[i, j] = func(M[colL[i]], M[colL[j]])
            cM[j, i] = func(M[colL[j]], M[colL[i]])
    return cM


def lagMatrix(X):
    """phase lag on a matrix"""
    #lagM = cross_funcM(X,timeLag)
    n = X.shape[1]
    lagM = np.zeros((n,n))
    for i, c in enumerate(X.columns):
        for j, r in enumerate(X.columns):
            lagM[i,j], _ = maxLag(X[c],X[r])
    return lagM


def interpMissing(y, isPlot=False):
    """interpolate missing"""
    name = None
    if isinstance(y, pd.Series):
        name = y.name
    y = np.array(y)
    y = y.astype(np.float)
    nans = [x != x for x in y]
    if sum(nans) <= 0.: return y
    if sum(nans) == len(nans): return np.linspace(0, 0, len(y))
    no_nan = [x == x for x in y]
    indices = np.arange(len(y))
    interp = np.interp(indices, indices[no_nan], y[no_nan])
    if name:
        interp = pd.Series(interp,name=name)
    if isPlot:
        plt.plot(interp)
        plt.plot(y)
        plt.show()
    return interp

    def nan_helper(y):
        return np.isnan(y, dtype=float), lambda z: z.nonzero()[0]

    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def interpMissingMatrix(X1, isPlot=False):
    """interpolate every single columns"""
    X = X1.copy()
    cL = list(range(X.shape[1]))
    if isinstance(X, pd.DataFrame):
        cL = X.columns
        for c in cL:
            X.loc[:,c] = interpMissing(X1[c].values)
    else:
        for c in cL:
            X[c] = interpMissing(X1[c])
    return X


def interpMissing2d(X, isPlot=False):
    """interpolate missing 2d"""
    X = np.array(X)
    X = X.astype(np.float)
    nans = X != X
    if nans.sum() <= 0.: return X
    no_nan = ~nans
    ix, iy = np.arange(X.shape[1]), np.arange(X.shape[0])
    xx, yy = np.meshgrid(ix, iy)
    Y = sp.interpolate.griddata((xx[no_nan], yy[no_nan])
                                , X[no_nan].ravel()
                                , (xx, yy), method='linear'
                                , fill_value=X[no_nan].mean())
    return Y


def missingDense(x, y, isDense=False, quant=0.7):
    """return missing dense points"""
    nans = np.isnan(y)
    countL = pd.Series(x[nans]).value_counts()
    countL = countL[countL > countL.quantile(quant)]
    if isDense:
        countL.sort_index(inplace=True)
        countL = pd.DataFrame(countL)
        countL.loc[:, "x"] = countL.index
        countL["ratio"] = [1] + [SequenceMatcher(None, x, y).ratio() for (x, y) in
                                 zip(countL['x'][1:], countL['x'][:-1])]
        countL = countL[countL['ratio'] >= 0.894737]
        return countL['x'].values
    return countL.index


def shift(y,lag=0):
    """shift and wrap array values"""
    if lag == 0 : return y
    shift = np.concatenate( (y[-lag:],y[:-lag]) )
    # shift = y.shift(lag)
    # shift.iloc[:lag] = y.iloc[-lag:].values
    return shift


def xCor(y1, y2):
    """compute a cross correlation between two signals"""
    xm1, xm2, ym1, ym2, xy = (0,) * 5
    N = len(y1)
    xM, yM = y1.mean(), y2.mean()
    xy = sum( (y1-xM) * (y2-yM) )
    xm2 = sum( (y1-xM)**2)
    ym2 = sum( (y2-yM)**2)
    xcor = (xy) / np.sqrt(xm2 * ym2)
    #xcor = y1.corr(y2.shift(lag))
    return xcor


def spectrum(y, isPlot=False, lab="spectrum"):
    """power spectrum of the signal"""
    yf = np.abs(sp.fft.fft(y / y.max())) ** 2
    freq = sp.fft.fftfreq(y.size, d=1. / 7.)
    yf = yf[freq > 0]
    if isPlot:
        plt.plot(yf,alpha=0.5,label=lab)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
    return yf


def synchrony(y1, y2, period=64, isPlot=False, labV = ["rolling median 1","rolling median 2"]):
    """syncronicity over a rolling window"""
    y1 = pd.Series(y1).interpolate()
    y2 = pd.Series(y2).interpolate()
    rolling_r = y1.rolling(window=period, center=True).corr(y2)
    if isPlot: 
        fig , ax = plt.subplots(2,1,figsize=(14,6),sharex=True)
        rolling_m1 = y1.rolling(window=period, center=True).median()
        ax[0].plot(rolling_m1,label=labV[0])
        rolling_m2 = y2.rolling(window=period, center=True).median()
        ax[0].plot(rolling_m2,label=labV[1])
        ax[0].set(xlabel='time',ylabel='values')
        ax[0].legend()
        rolling_r.plot(ax=ax[1])
        ax[1].set(xlabel='time',ylabel='Pearson r')
    return rolling_r


def decayM(M,period=64):
    """compute cross correlation between matrix columns"""
    #cM = cross_funcM(M, xCor)
    def func(y1,y2):
        res, xcor = xcorDecay(y1,y2,period=64)
        return sum(xcor)/len(xcor)
    
    colL = [x for x in M.columns]
    N = len(colL)
    cM = np.zeros((N, N))
    for i in range(N):
        cM[i, i] = func(M[colL[i]], M[colL[i]])
        for j in range(i + 1, N):
            cM[i, j] = func(M[colL[i]], M[colL[j]])
            cM[j, i] = func(M[colL[j]], M[colL[i]])
    return cM


def autCorM(M):
    """compute autocorrelation function on matrix columns"""
    colL = [x for x in M.columns]
    acM = pd.DataFrame()
    for i in colL:
        r, q, p = sm.tsa.acf(M[i], qstat=True)
        acM[i] = r
    return acM


def pautCorM(M):
    """compute partial auto correlation function on matrix columns"""
    colL = [x for x in M.columns]
    acM = pd.DataFrame()
    for i in colL:
        r = tsa.stattools.pacf(M[i], nlags=20, method='ols')
        acM[i] = r
    return acM


def xcorM(M, L):
    """compute cross correlation between matrices"""
    colL = [x for x in M.columns]
    acM = pd.DataFrame()
    for i in colL:
        r = np.correlate(M[i], L[i], "full")
        acM[i] = r
    return acM


def cross_funcM(M, func):
    """apply a function across all pairs"""
    M = np.array(M)
    d = np.zeros((M.shape[1], M.shape[1]))
    comb = np.array(list(itertools.combinations(range(M.shape[1]), 2)))
    crossL = []
    for c1, c2 in zip(comb[:, 0], comb[:, 1]):
        x, y = M[:, c1], M[:, c2]
        score = func(x, y)
        crossL.append(score)
    d[comb[:, 0], comb[:, 1]] = crossL
    d[comb[:, 1], comb[:, 0]] = crossL
    return d


def cross_corrM(M):
    """compute cross entropy between fields"""
    def func(x, y):
        return sp.stats.pearsonr(x, y)[0]
    return cross_funcM(M, func)


def cross_entropyM(M):
    """compute cross entropy between fields"""
    return cross_funcM(M, sk.metrics.mutual_info_score)


def applyNextM(M):
    """apply a function between neighboring columns"""
    new = np.dstack((M[:, :-1], M[:, 1:]))
    M1 = np.apply_along_axis(np.sum, 2, new)
    return M1


def decayM1(M):
    """compute decay exponent from auto correlation"""
    def ser_exp(x, t, p):
        exp1 = np.exp(-x[0]*t)
        return x[1]*exp1

    colL = [x for x in M.columns]
    acM = pd.DataFrame()
    for i in colL:
        r = M[i]
        X = np.array(range(0, r.size, 7))
        popt, pcov = sp.optimize.curve_fit(ser_exp, np.array(range(0, 6)), r[0:6] - min(r), p0=(1))
        popt1, pcov1 = sp.optimize.curve_fit(ser_exp, np.array(range(0, r.size, 7)), r[X], p0=(1))
        acM[i] = np.array([popt[0], pcov[0][0], popt1[0], pcov1[0][0]])
    return acM


def linReg(x, y, isLog=False):
    """perform a linear regression"""
    x = np.array(x.copy())
    y = np.array(y.copy())
    if isLog:
        x = np.log(x)
        y = np.log(y)
    regr = linear_model.LinearRegression()
    X = np.reshape(x, (-1, 1))
    regr.fit(X, y)
    return regr.predict(X)


def gaussM(M):
    """Gaussian interpolation on matrix columns"""
    colL = [x for x in M.columns]
    acM = pd.DataFrame()

    def fun(x, t):
        return x[2] * np.exp(-pow((t - x[0]), 2) * x[1])

    def fun_min(x, t, y):
        return fun(x, t) - y

    x0 = [float(M.shape[0] / 2), 1., 1.]
    for i in colL:
        t = [float(x) for x in M.index]
        y = [x for x in M[i]]
        res_lsq = least_squares(fun_min, x0, args=(t, y))
        y = [fun(res_lsq[0], x) for x in t]
        acM[i] = y  # res_lsq[0]
        x0 = res_lsq[0]
    return acM


def matNN(M):
    """prepare cross correlation between matrix entries for narest neighbors"""
    # in progress
    colL = [x for x in M.columns]
    N = len(colL)
    Niter = 100
    for i in range(Niter):
        col1 = int(random.random() * N)
        col2 = int(random.random() * N)
        if (col1 == col2):
            next
        colD = col1 - col2
        cM = M
    for i in range(N):
        i1 = i + 1 if i < N else 0
        i2 = i - 1 if i > 0 else N
        for j in range(i + 1, N):
            j1 = j + 1 if j < N else 0
            j2 = j - 1 if j > 0 else N
            cM[i, j] = corS(webH[colL[i]], webH[colL[j]])
            cM[j, i] = corS(webH[colL[i]], webH[colL[j]])
    return cM


def serTestStat(timeseries):
    """Standard tests on time series"""
    from statsmodels.tsa.stattools import adfuller
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def getCurveStat(y, isPlot=False):
    """calculate autocorrelation, power spectrum"""
    n = len(y)
    variance = y.var()
    ym1 = (y - y.mean())
    ym = (y - y.mean()) / y.mean()
    y3 = ym * ym * ym
    r = np.correlate(ym1, ym1, mode='full')[-n:]
    # isClose = np.allclose(r, np.array([(ym1[:n - k] * ym1[-(n - k):]).sum() for k in range(n)]))
    autocor = r / (variance * (np.arange(n, 0, -1)))
    x = np.array([x for x in range(0, n)])
    xp = [i for (i, x) in enumerate(autocor) if x > 0]
    yp = [x for (i, x) in enumerate(autocor) if x > 0]

    def expD(x, b):
        return np.exp(-b * x)

    def linD(x, b, c):
        return -b * x + c

    nInterp = min(48, len(y))
    popt, pcov = sp.optimize.curve_fit(expD, xp[:nInterp], yp[:nInterp], p0=[0.1])
    #yf = np.abs(sp.fftpack.fft(y / y.max())) ** 2
    yf = np.abs(sp.fft.fft(y / y.max())) ** 2
    freq = sp.fft.fftfreq(y.size, d=1. / 7.)
    yf = yf[freq > 0]
    freq = freq[freq > 0]
    xl = np.array([np.log(x) for x in range(1, len(yf) + 1)])
    yl = np.array([np.log(abs(x)) for x in yf])
    fopt, fcov = sp.optimize.curve_fit(linD, xl[:nInterp], yl[:nInterp], p0=[-1.0, 4.])
    chiSq = sp.stats.chisquare(y)[1]
    f_w = yf[7] if len(yf) > 7 else 0.
    f_biw = yf[14] if len(yf) > 14 else 0.
    if isPlot:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle("statistical properties of time series")
        ax[0][0].set_title("distribution")
        ax[0][0].hist(ym, bins=20, label="histogram")
        ax[0][0].set_xlabel("mean deviation")
        ax[0][0].set_ylabel("counts")
        ax[0][0].set_yscale('log')
        ax[0][0].legend()
        ax[0][1].set_title("autocorrelation")
        ax[0][1].plot(autocor, label="autocor")
        ax[0][1].plot(xp, yp, label="autocor positive")
        ax[0][1].plot(expD(x[:nInterp], *popt), label="interpolated")
        ax[0][1].set_xlabel("lag (deltat)")
        ax[0][1].set_ylabel("autocorrelation")
        ax[0][1].set_xscale('log')
        ax[0][1].set_yscale('log')
        ax[0][1].legend()
        ax[1][0].set_title("power spectrum interpolation")
        ax[1][0].plot(xl, yl, label="power spectrum")
        ax[1][0].plot(xl, linD(xl, *fopt), label="interpolation")
        ax[1][0].set_xlabel("log lag (deltat)")
        ax[1][0].set_ylabel("log power")
        ax[1][0].legend()
        ax[1][1].set_title("autocorrelation decay")
        ax[1][1].set_title("harmonics")
        ax[1][1].plot(yf, label="power spectrum")
        # ax[1][1].axvline(x=7, color='b', linestyle='-', label="1 week")
        # ax[1][1].axvline(x=14, color='b', linestyle='-', label="2 week")
        # ax[1][1].axvline(x=21, color='b', linestyle='-', label="3 week")
        # ax[1][1].axvline(x=28, color='b', linestyle='-', label="4 week")
        ax[1][1].set_xlabel("lag (deltat)")
        ax[1][1].set_ylabel("power")
        ax[1][1].set_yscale('log')
        ax[1][1].legend()
        plt.show()
        
    return {"daily_visit": y.mean(), "auto_decay": popt[0], "noise_decay": fopt[0], "harm_week": f_w,
            "harm_biweek": f_biw, "y_var": y.var() / y.mean(), "y_skew": y3.sum(), "chi2": chiSq, "n_sample": n}


def fromMultivariate(X, mode="pca"):
    """from a multivariate matrix X (d,n) extract the most significant component or the pdf
    output:
       density or main component, covariance
    input:
       X = (d,n) numpy array
       mode = 'pca' (principal component analysis), 'ica' (independent component analysis), 'normal' pdf, 't-distribution' multivariate t-distribution, 'log-t' log multivariate t density
    """
    [n, d] = X.shape
    df = d - 1
    mu = X.mean(axis=0)
    Xm = X - mu
    S = np.cov(X.T)
    det = np.linalg.det(S)
    if abs(det) < 1e-24:
        print('singular matrix, reduce dimensionality')
        return np.array([0])
    inv = np.linalg.inv(S)
    if mode == 'normal':
        var = sp.stats.multivariate_normal(mean=mu, cov=S)
        pdf = var.pdf(X)
        return pdf, S

    if mode == 'norm-pdf':
        norm = 1.0 / (math.pow((2 * math.pi), float(d) / 2) * math.pow(det, 1.0 / 2))
        inv2 = -0.5 * np.dot(np.dot(Xm, inv).T, Xm)
        result = norm * math.pow(math.e, inv2)
        part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (pow(det, .5)))
        part2 = 0.5 * ((Xm).dot(inv).T.dot((Xm)))
        return float(part1 * np.exp(part2))
    x = np.random.randn(3, 3)
    mu = x.mean(axis=1)
    cov = np.cov(x)

    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    print(float(part1 * np.exp(part2)))

    if mode == 't-distribution':
        Num = math.gamma(1. * (d + df) * .5)
        Denom = math.gamma(1. * df * .5) * pow(df * math.pi, 1. * d * .5) * pow(det, .5)
        inv2 = 1 + (1. / df) * np.dot(np.dot(Xm, inv).T, Xm)
        Denom = Denom * pow(inv, 1. * (d + df) / 2)
        pdf = 1. * Num / Denom
        return pdf, S

    if mode == 'log-t':
        V = df * S
        V_inv = np.linalg.inv(V)
        (sign, logdet) = np.linalg.slogdet(np.pi * V)
        logz = -math.gamma(df / 2.0 + d / 2.0) + math.gamma(df / 2.0) + 0.5 * logdet
        logp = -0.5 * (df + d) * np.log(1 + np.sum(np.dot(Xm, V_inv) * Xm, axis=1))
        logp = logp - logz
        return logp, S

    if mode == 'ica':
        ica = FastICA(n_components=d)
        iS = ica.fit_transform(X)  # Reconstruct signals
        iA = ica.mixing_  # Get estimated mixing matrix
        iC = ica.components_
        return iS[:, 1], iA

    if mode == 'pca':
        pca = PCA(n_components=d)
        H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
        SH = np.cov(H.T)
        return H[:, 1], SH

    if mode == 'hotelling':
        pc = PCA(n_components=d).fit(X)
        coeff = pc.Wt
        x = pc.transform(X).T
        pc = mlab.PCA(X)
        coeff = pc.Wt
        x = pc.a.T
        s = pc.s
        cov = coeff.T.dot(np.diag(s)).dot(coeff) / (x.shape[1] - 1)
        w = np.linalg.solve(cov, x)
        t2 = (x * w).sum(axis=0)
        ed = np.sqrt(sp.stats.chi2.ppf(0.95, 2))
        np.linalg.cholesky(s)
        return t2, cov

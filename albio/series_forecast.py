import datetime
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as sco
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq as least_squares

import albio.algo_holtwinters as ht
import albio.series_stat as a_s
import albio.forecast as a_f

    
#-------------------------series-class----------------------------

def standardConf():
    return {"obs_time":0,"decay":[1],"hist_adj": 0.8,"model": "holt"
            ,"res":[14,0.03,0.01,0.02,0.15,0.12,0.13,0.11,0.22]
            ,"holt": [5,0.2,0.1,0.05,0.8]
            ,"poly": [-0.26,-0.043,-1.24,0.0024,-1.6e-06,3.6e-10]
            ,"response": [1.37,1.09,-0.57,0.023,0.110]            
            ,"autocor": [0.095,-1.01,-0.40,0.15,8.26,1.0,1.0,1.0,1.0]
            ,"freq": [10.38,5.19]
            ,"arma": [4,0,0]
            ,"sin": [0.,1.,0.03,0.31,2.80,1.0,1.0,1.0,1.0] 
    }

class series:
    """statistical properties and forecast of series"""

    def __init__(y, t=[None], conf=None, isPlot=False):
        """analyze time series"""
        ny = len(y)
        if t[0] == None:
            t0 = datetime.datetime.today()-datetime.timedelta(ny)
            t = [t0+datetime.timedelta(x) for x in range(ny)]
        if conf == None:
            conf = standardConf()
        week = [x.isocalendar()[1] for x in t]
        month = [x.month for x in t]
        year = [x.isocalendar()[0] for x in t]
        ser = pd.DataFrame({"y":y,"t":t,"week":week,"month":month,"year":year})
        ser.loc[:,"ts"] = [x.timestamp() for x in t]
        sWeek = ser[['ts','y','week']].groupby(["week"]).agg(np.mean)
        s = ser.iloc[0]
        s1 = pd.DataFrame({'ts':s['ts']-1,'y':s['y']},index=[s['week']])
        s = ser.iloc[-1]
        s2 = pd.DataFrame({'ts':s['ts']+1,'y':s['y']},index=[s['week']])
        sWeek = pd.concat([s1,sWeek,s2])
        sMont = ser[['ts','y','month']].groupby(["month"]).agg(np.mean)
        sYear = ser[['ts','y','year']].groupby(["year"]).agg(np.mean)
        ser = ser.merge(sWeek[['y']],on="week" ,how="left",suffixes=["","_week"])
        ser = ser.merge(sMont[['y']],on="month",how="left",suffixes=["","_month"])
        ser = ser.merge(sYear[['y']],on="year" ,how="left",suffixes=["","_year"])
        ser.loc[:,"roll"] = a_s.serSmooth(ser['y'],4,11)
        ser.loc[:,"e_av"] = ser['y'].ewm(halflife=1).mean()
        s2 = pd.DataFrame({"week":s['week'],'ts':s['ts'],'y':s['y']})
        ser['inte'] = sp.interpolate.interp1d(sWeek['ts'],sWeek['y'],kind="cubic")(ser['ts'])
        ser['deri'] = ser['y'] - ser['y'].shift()
        ser['diff'] = ser['y'] - ser['e_av']
        ser['rema'] = ser['y'] - ser['roll']
        ser['hist'] = ser['y'] - ser['inte']
        ser['adju'] = ( (ser['hist']-ser['hist'].min())/(ser['hist'].max()-ser['hist'].min()) + .5)*conf['hist_adj']
        ser.drop(columns={"week","month","year"},inplace=True)
        ser.replace(float('nan'),0.,inplace=True)
        self.ser = ser
        self.sWeek = sWeek
        if isPlot:
            plt.bar( ser['t'],ser['y_year'],label="yearly",alpha=0.1) 
            plt.bar( ser['t'],ser['y_month'],label="monthly",alpha=0.1)
            plt.bar( ser['t'],ser['y_week'],label="weekly",alpha=0.1)
            plt.plot(ser['t'],ser['y'],label="series",linewidth=2)
            plt.plot(ser['t'],ser['roll'],label="rolling average",linewidth=1)
            plt.plot(ser['t'],ser['e_av'],label="average",linewidth=1)
            plt.plot(ser['t'],ser['inte'],label="interpolate week",linewidth=1)
            plt.legend()
            plt.xticks(rotation=15)
            plt.show()
            plt.plot(ser['t'],ser['deri'],label="derivative",linewidth=1)
            plt.plot(ser['t'],ser['diff'],label="remainder exp",linewidth=1)
            plt.plot(ser['t'],ser['rema'],label="remainder roll",linewidth=1)
            plt.plot(ser['t'],ser['hist'],label="historical week",linewidth=1)
            plt.legend()
            plt.xticks(rotation=15)
            plt.show()


    def trendFluct(self, n_ahead=14):
        """predict the signal decomposing the trend from the fluctuations"""
        ser = self.ser
        t1, y1, pol = a_f.trendFluct(ser['t'],ser['t'],ser['y'],n=14,conf=conf)
        return t1, y1, pol



##---------------------------series-preprocessing------------------------



##---------------------------forecast-methods----------------------------

def serLsq(sDay, nAhead, x0, hWeek):
    nFit = sDay.shape[0]  # if int(x0['obs_time']) <= 14 else int(x0['obs_time'])
    predS, x0 = getHistory(sDay, nAhead, x0, hWeek)
    predS = predS.tail(nFit + nAhead)
    freqP = x0['freq']
    res_lsq = least_squares(ser_sin_min, x0['lsq'], args=(sDay.t, sDay.stat, x0['freq']))  # loss='soft_l1',f_scale=0.1,
    x0['lsq'] = [x for x in res_lsq[0]]
    predS['lsq'] = ser_sin(res_lsq[0], predS.t, x0['freq'])  # fun(res_robust.x,t_test)
    sDay['resid'] = sDay['y'] - ser_sin(res_lsq[0], sDay.t, x0['freq']) / sDay['hist'] - ser_poly(x0['poly'], sDay[
        't'])  # lm.predict(sDay)
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean()) ** 2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum() / sDay['y'].tail(x0['res'][0]).sum()
    # predS['pred'] = (predS['lsq'] + lm.predict(predS))*predS['hist']*x0['hist_adj']
    predS['pred'] = ser_sin(res_lsq[0], predS.t, x0['freq']) * sDay['hist'] + ser_poly(x0['poly'], predS['t'])
    predS = predS.drop(predS.index[0])
    return predS, x0


def bestArima(sDay, nAhead, x0, hWeek):
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10], '%Y-%m-%d') for x in dta.index]
    t_line = [float(calendar.timegm(x.utctimetuple())) / 1000000 for x in dta.index]
    t_line1 = [float(calendar.timegm(x.utctimetuple())) / 1000000 for x in hWeek.index]
    sExog = pd.DataFrame({'y': sp.interpolate.interp1d(t_line1, hWeek.y, kind="cubic")(t_line)})
    grid = (slice(1, 4, 1), slice(1, 2, 1), slice(1, 3, 1))

    def objfunc(order, endog, exog):
        fit = sm.tsa.ARIMA(endog, order).fit(trend="c", method='css-mle', exog=exog)
        return fit.aic

    par = sco.brute(objfunc, grid, args=(dta, sExog), finish=None)
    return par


def serArma(sDay, nAhead, x0, hWeek):
    predS, x0 = getHistory(sDay, nAhead, x0, hWeek)
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10], '%Y-%m-%d') for x in dta.index]
    sDay.index = dta.index
    t_line = [float(calendar.timegm(x.utctimetuple())) / 1000000 for x in dta.index]
    t_line1 = [float(calendar.timegm(x.utctimetuple())) / 1000000 for x in hWeek.index]
    sExog = pd.DataFrame({'y': sp.interpolate.interp1d(t_line1, hWeek.y, kind="cubic")(t_line)})
    # par = bestArima(dta,sExog)
    sExog.index = dta.index
    result = sm.tsa.ARIMA(dta, (x0['arma'][0], x0['arma'][1], x0['arma'][2])).fit(trend="c", method='css-mle',
                                                                                  exog=sExog)
    predT = [str(dta.index[0])[0:10], str(dta.index[len(dta) - 1])[0:10]]
    histS = pd.DataFrame({'pred': result.predict(start=predT[0], end=predT[1])})
    # predT = [str(dta.index[0])[0:10],str(dta.index[len(dta)-1]+datetime.timedelta(days=nAhead))[0:10]]
    predS = predS[0:(len(dta) - 1)]
    predS['pred'] = result.predict(start=predT[0], end=predT[1])
    # predS = pd.DataFrame({'pred':result.predict(start=predT[0],end=predT[1])})
    # predS.index = [dta.index[0]+datetime.timedelta(days=x) for x in range(0,len(dta)+nAhead)]
    # predS['t'] = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in predS.index]
    # predS['hist'] = sp.interpolate.interp1d(t_line1,hWeek.y,kind="cubic")(predS['t'])
    # predS['hist'] = predS['hist']/predS['hist'].mean()
    # predS['pred'] = predS['pred']*predS['hist']
    predS['trend'] = ser_poly(x0['poly'], predS.t)
    predS['y'] = sDay['y']
    predS['pred'] = (predS['pred'] * predS['hist'] + predS['trend'])
    x0['response'] = [x for x in result.params]
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean()) ** 2
    x0['res'][1:2] = [rSquare.sum(), rSquare.sum() / sDay['y'].tail(x0['res'][0]).sum()]
    return predS, x0
    # plt.plot(dta,'-k',label="series")
    # plt.plot(sExog,label="exo")
    # plt.plot(hWeek.y,label="hist")
    # plt.plot(predS,'-b',label="pred")
    # plt.legend()
    # plt.show()
    # steps = 1
    # tsa.arima_model._arma_predict_out_of_sample(res.params,steps,res.resid,res.k_ar,res.k_ma,res.k_trend,res.k_exog,endog=dta, exog=None, start=len(dta))


def SerBayes(sDay, nAhead, x0, hWeek):
    import pydlm
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10], '%Y-%m-%d') for x in dta.index]
    t_line = [float(calendar.timegm(x.utctimetuple())) / 1000000 for x in dta.index]
    dta.index = t_line
    model = pydlm.dlm(dta)
    model = model + pydlm.trend(degree=1, discount=0.98, name='a', w=10.0)
    model = model + pydlm.dynamic(features=[[v] for v in t_line], discount=1, name='b', w=10.0)
    model = model + pydlm.autoReg(degree=3, data=dta.values, name='ar3', w=1.0)
    allStates = model.getLatentState(filterType='forwardFilter')
    model.evolveMode('independent')
    model.noisePrior(2.0)
    model.fit()
    model.plot()
    model.turnOff('predict')
    model.plotCoef(name='a')
    model.plotCoef(name='b')
    model.plotCoef(name='ar3')


def serHolt(sDay, nAhead, x0, hWeek):
    predS, x0 = getHistory(sDay, nAhead, x0, hWeek)
    Y = [x for x in sDay.y]
    ##Yht, alpha, beta, gamma, rmse = ht.additive([x for x in sDay.y],int(x0[0]),nAhead,x0[1],x0[2],x0[3])
    nAv = int(x0['holt'][0]) if int(x0['holt'][0]) > 1 else 5
    Yht, alpha, beta, gamma, rmse = ht.additive([x for x in sDay.y], x0['holt'][0], nAhead, x0['holt'][1],
                                                x0['holt'][2], x0['holt'][3])
    sDay['resid'] = sDay['y'] - Yht[0:sDay.shape[0]]
    x0['holt'] = [x0['holt'][0], alpha, beta, gamma, rmse]
    nLin = sDay.shape[0] + nAhead
    t_test = np.linspace(sDay['t'][0], sDay['t'][sDay.shape[0] - 1] + sDay.t[nAhead] - sDay.t[0], nLin)
    # predS = pd.DataFrame({'t':t_test},index=[sDay.index[0]+datetime.timedelta(days=x) for x in range(nLin)])
    predS['pred'] = Yht
    # predS['hist'] = sp.interpolate.interp1d(hWeek.t,hWeek.y,kind="cubic")(predS['t'])
    # predS['hist'] = predS['hist']/predS['hist'].mean()
    # predS['pred'] = predS['pred']*predS['hist']*x0['hist_adj']
    # predS['trend'] = ser_poly(x0['poly'],predS.t)
    # predS['trend'].ix[0:(sDay.shape[0]-nFit)] = predS['trend'][sDay.shape[0]-nFit]
    predS['lsq'] = 0
    predS['y'] = sDay['y']
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean()) ** 2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum() / sDay['y'].tail(x0['res'][0]).sum()
    return predS, x0


def serAuto(sDay, nAhead, x0, hWeek):
    predS, x0 = getHistory(sDay, nAhead, x0, hWeek)
    todayD = datetime.datetime.today()
    todayD = todayD.replace(hour=0, minute=0, second=0, microsecond=0)
    dta = pd.DataFrame({'y': sDay.y})
    dta['day'] = sDay.index.weekday
    phase = dta.head(int(x0['obs_time'])).groupby(['day']).mean()
    phase['std'] = dta.groupby(['day']).std()['y']
    phase = phase.sort_values(['y'], ascending=False)
    phase['csum'] = phase['y'].cumsum() / phase['y'].sum()
    phaseN = phase.index[0] - todayD.weekday()
    r, q, p = sm.tsa.acf(sDay['y'].tail(phaseN + int(x0['obs_time'])).squeeze(), qstat=True)
    popt, pcov = curve_fit(ser_exp, np.array(range(0, 6)), r[0:6] - min(r), p0=(x0['decay'][0]))
    X = np.array(range(0, r.size, 7))
    popt1, pcov1 = curve_fit(ser_exp, X, r[X], p0=(x0['decay'][0]))
    autD = pd.DataFrame({'r': r, 'exp': ser_exp(range(0, r.size), popt), 'exp1': ser_exp(range(0, r.size), popt1)})
    x0['decay'] = [x for x in popt]
    wN = 0
    sY = np.random.normal(phase['y'].head(1), dta.y.std())
    for i in predS.index:
        wN = 6 - np.abs(phase.index[0] - i.weekday())
        wN = wN + 1 if wN < 6 else 0
        if (wN == 0):
            sY = np.random.normal(phase['y'].head(1), dta.y.std() / 2)
        sY = sY * (1 + predS['hist'][i] * x0['hist_adj'])
        predS.loc[i, 'pred'] = sY * ser_exp(float(wN), popt)

    predS['pred'] = serSmooth(predS['pred'], 16, 5)
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    freqP = x0['freq']
    res_lsq = least_squares(ser_fun_min, x0['lsq'], args=(sDay['t'], sDay['resid'], x0['freq']))
    predS['lsq'] = ser_sin(res_lsq[0], predS['t'], x0['freq'])  # fun(res_robust.x,t_test)
    x0['lsq'][0:res_lsq[0].size] = res_lsq[0]
    predS['pred2'] = predS['pred']
    predS['pred'] = predS['pred'] + predS['lsq']
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean()) ** 2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum() / sDay['y'].tail(x0['res'][0]).sum()
    # sDay.to_csv('tmpAuto1.csv')
    # predS.to_csv('tmpAuto2.csv')
    # autD.to_csv('tmpAuto3.csv')
    return predS, x0

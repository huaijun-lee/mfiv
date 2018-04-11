#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-14 10:13:46
# @Author  : Li Huaijun (huaijun000@gmail.com)

'''
处理股指期权数据
'''

import os
from os import path
import pandas as pd
import numpy as np
from scipy import stats, interpolate
from six.moves import cPickle as pkl
from math import exp, pow, sqrt, log
from py_vollib.black_scholes import implied_volatility as BSImpVol
import matplotlib.pyplot as plt

from RealVol import RealiedVol


def process_price(data_file):
    '''
    提取沪深300交易数据，删除每列的中文说明及单位标注，#同时对数据按交易时间排序,
    并计算交易时间距到期的天数
    输入为文件所在路径
    '''
    # os.remove(path.join(temp_path, 'pricing_data.plk'))
    try:
        newdoc = pkl.load(open(path.join(temp_path, 'pricing_data.plk'), 'rb'))
    except FileNotFoundError:
        doc = pd.read_excel(data_file)
        newdoc = doc[2:]
        newdoc = newdoc[newdoc['UnderlyingSecuritySymbol'] == '000300']
        newdoc['RemainingDay'] = newdoc['RemainingTerm'].map(lambda x: round(x * 365))
        # newdoc = newdoc.sort_values(by='TradingDate')
        newdoc = newdoc.fillna(0)
        newdoc.index = range(len(newdoc))
        pkl.dump(newdoc, open(path.join(temp_path, 'pricing_data.plk'), 'wb'))
    return newdoc


def distribution(doc):
    '''
    计算数据中按照时间的数据量的分布，同时计算该天行权价的最大值与最小值
    '''
    den = pd.DataFrame(doc.groupby('TradingDate')['TradingDate'].count())
    strikemax = pd.DataFrame(doc.groupby('TradingDate')['StrikePrice'].max())
    strikemin = pd.DataFrame(doc.groupby('TradingDate')['StrikePrice'].min())
    strikemin.columns = ['min']
    strikemax.columns = ['max']
    den.columns = ['count']
    price_df = den.join(strikemax).join(strikemin)
    return price_df


def CalculationIV(StrikePrice, ClosePrice, UnderlyingScrtClose, TimeValue, deltak):
    '''
    无模型隐含波动率单点数值计算
    '''
    singlevalue = (ClosePrice * TimeValue - max(
        UnderlyingScrtClose * TimeValue - StrikePrice, 0)) / pow(StrikePrice, 2) * deltak
    return singlevalue


def BSMCallValue(S0, K, T, r, sigma):
    """Black-sckholes 期权定价公式
    S0:在时间点t时刻标的物价格水平
    sigma：标的物固定波动率
    K：期权行权价格
    T：期权到期日
    r：固定无风险短期利率
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    Nd1 = stats.norm.cdf(d1, 0.0, 1.0)
    Nd2 = stats.norm.cdf(d2, 0.0, 1.0)
    value = S0 * Nd1 - exp(-r * T) * K * Nd2
    return value


def Implied_Vol(price, S, K, RemainingTerm, RisklessRate):
    try:
        val = BSImpVol.implied_volatility(
                                    price, S, K, RemainingTerm, RisklessRate, 'c')
    except Exception as e:
        # raise e
        val = 0
    return val


def ImpliedVarianceUtils(Groupeddata):
    '''
    计算单个到期时间的无模型隐含波动率及数据的预处理,
    无模型隐含波动率的计算暂为不考虑截断误差及不连续误差的情况
    '''
    def is_AP(sequ):
        '''
        判断序列是否为等差序列,及计算等差值
        '''
        ln = len(sequ)
        if ln == 1:
            return True, 50
        elif ln < 1:
            return False, 0
        else:
            n, m = min(sequ), max(sequ)
            dif = (m - n) // (ln - 1)
            rng = range(n, m, dif)
            if len(rng) != ln:
                return False, dif
            for i in sequ:
                if i not in rng:
                    return False, dif
            return True, dif
    RisklessRate = Groupeddata['RisklessRate'].max() / 100
    RemainingTerm = Groupeddata['RemainingTerm'].max()
    TimeValue = exp(RisklessRate * RemainingTerm)
    UnderlyingScrtClose = Groupeddata['UnderlyingScrtClose'].max()
    calldata = Groupeddata[Groupeddata['CallOrPut'] == 'C']
    strike_close_dict = dict(zip(calldata['StrikePrice'], calldata['ClosePrice']))
    StrikeSet = set(calldata['StrikePrice'])
    isap, deltak = is_AP(StrikeSet)
    if isap:
        ValueList = list(map(lambda x: CalculationIV(x, strike_close_dict.get(x),
                                                     UnderlyingScrtClose, TimeValue, deltak),
                             StrikeSet))
    elif deltak == 0:
        return 0.0001
    else:
        BSImpliedVol = list(map(lambda price, S, K:
                                Implied_Vol(
                                    price, S, K, RemainingTerm, RisklessRate),
                                calldata['ClosePrice'], calldata['UnderlyingScrtClose'],
                                calldata['StrikePrice']))
        try:
            tck = interpolate.splrep(calldata['StrikePrice'], BSImpliedVol,
                                     k=min(3, len(calldata['StrikePrice']) - 1), s=0)
            Strikelist = range(min(StrikeSet), max(StrikeSet), 50)
            for i in Strikelist:
                if i not in StrikeSet:
                    iv = max(interpolate.splev(i, tck, der=0), 0.0001)
                    try:
                        price = BSMCallValue(UnderlyingScrtClose, iv, i,
                                             RemainingTerm, RisklessRate)
                        strike_close_dict[i] = price
                    except ValueError as e:
                        print(iv)
                        print(BSImpliedVol)
                        print(calldata['StrikePrice'])
                        raise e
                    strike_close_dict[i] = price
        except TypeError as e:
            print(calldata['StrikePrice'])
            raise e
        ValueList = list(map(lambda x: CalculationIV(x, strike_close_dict.get(x),
                                                     UnderlyingScrtClose, TimeValue, deltak),
                             StrikeSet))
    mfiv_val = sqrt(2 * sum(ValueList))
    return mfiv_val


def ThirtyDaysIV(GData):
    '''
    由两个到期的期限的无模型波动率计算30天到期的无模型波动率
    '''
    val = (GData['variance'][0] * GData['RemainingDayC'][0] *
           (GData['RemainingDayC'][1] - 30) + GData['variance'][1] *
           GData['RemainingDayC'][1] * (30 - GData['RemainingDayC'][0]))
    return val / ((GData['RemainingDayC'][1] - GData['RemainingDayC'][0]) * 30)


def DataFilter(doc):
    '''
    对数据进行筛选，筛选规则有：
    1. 选取距离到期日大于7天且小于81天的交易数据
    2. 选取Call 的delta大于0.05且小于0.5，而Put 的delta 大于 -0.5 且小于 -0.05
    '''
    def DeltaSelect(CallOrPut, Delta):
        if CallOrPut == 'C':
            if Delta > 0.05 and Delta < 0.5:
                return True
            else:
                return False
        else:
            if Delta > -0.5 and Delta < -0.05:
                return True
            else:
                return False
    remainingselectBool = np.array(list(map(lambda x, y: x > 7 and y < 81,
                                            doc['RemainingDay'], doc['RemainingDay'])))
    deltaselectBool = np.array(list(map(lambda x, y: DeltaSelect(x, y),
                                        doc['CallOrPut'], doc['Delta'])))
    return remainingselectBool & deltaselectBool


def MFIV(doc, withio=False):
    '''
    计算数据中的无模型隐含波动率及波动率风险溢价，
    输入数据为含有具体到期天数的DataFrame对象
    输出为日期与隐含波动率、波动率风险溢价相对应的DataFrame对象
    '''
    isselect = DataFilter(doc)
    selectdata = doc[isselect].copy()
    Groupeddata = selectdata.groupby(['TradingDate', 'RemainingDay'])
    TwoDaysIV = pd.DataFrame(Groupeddata.apply(ImpliedVarianceUtils))
    TwoDaysIV.columns = ['variance']
    TwoDaysIV['TradingDateC'] = list(map(lambda x: x[0], TwoDaysIV.index))
    TwoDaysIV['RemainingDayC'] = list(map(lambda x: x[1], TwoDaysIV.index))
    GData = TwoDaysIV.groupby('TradingDateC')
    MFIVData = pd.DataFrame(GData.apply(ThirtyDaysIV))
    MFIVData.columns = ['MFIV']
    RealVolDF = RealiedVol(selectdata)
    VRPData = RealVolDF.join(MFIVData, on='Date')
    VRPData['VRP'] = list(map(lambda x, y: x - y,
                              VRPData['RealVol'], VRPData['MFIV']))
    if withio:
        selectdata.to_excel(path.join(out_path, 'selectdata.xlsx'))
        TwoDaysIV.to_excel(path.join(out_path, 'MidResultdata.xlsx'))
        MFIVData.to_excel(path.join(out_path, 'MFIVresultdata.xlsx'))
        VRPData.to_excel(path.join(out_path, 'VRPresultdata.xlsx'))
    return VRPData


def mfiv_plot(MFIVData):
    '''
    画时序图
    '''
    MFIVData = MFIVData.sort_values(by='Date')
    MFIVData.index = list(map(pd.Timestamp, MFIVData['Date']))
    del MFIVData['Date']
    MFIV = MFIVData['MFIV']
    # plt.figure(dpi=600)
    MFIV.plot()
    plt.title('无模型隐含波动率时序图')
    plt.savefig(path.join(out_path, 'mfivtsplot.png'), dpi=600)
    plt.figure()
    MFIV.plot.kde()
    plt.title('无模型隐含波动率的核密度曲线图')
    plt.savefig(path.join(out_path, 'mfivkdeplot.png'), dpi=600)
    # plt.show()


rootdir = os.getcwd()
data_path = path.join(rootdir, 'data')
temp_path = path.join(rootdir, 'temp')
out_path = path.join(rootdir, 'output')
if not path.exists(temp_path):
    os.mkdir(temp_path)
if not path.exists(data_path):
    os.mkdir(data_path)
if not path.exists(out_path):
    os.mkdir(out_path)
if __name__ == '__main__':
    # data_file = path.join(data_path, 'IO_PricingParameter.xlsx')
    # doc = process_price(data_file)
    # VPRData = MFIV(doc, True)
    # pkl.dump(VPRData, open(path.join(temp_path, 'VPR_data.plk'), 'wb'))
    VPRData = pkl.load(open(path.join(temp_path, 'VPR_data.plk'), 'rb'))
    # MFIVData.to_excel(path.join(out_path, 'MFIVresultdata.xlsx'))
    mfiv_plot(VPRData)

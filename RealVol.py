#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-22 11:17:58
# @Author  : Li Huaijun (huaijun000@gmail.com)
import numpy as np
import pandas as pd
from math import sqrt, log


def RealiedVol(doc):
    '''
    计算标的证券的已实现波动率：

    输入为含有交易日期及证券收盘价的DataFrame格式数据
    返回交易日期当天、周期为30日的年化已实现波动率，其中一年交易日定为250个交易日
    结构为DataFrame 格式
    '''
    LogClosePrice = doc.groupby('TradingDate')[
        'UnderlyingScrtClose'].apply(lambda x: log(x.mean()))
    diffprice = LogClosePrice.diff(periods=1)
    DiffPriceSer = diffprice.dropna(axis=0).sort_index()
    Pricedf = pd.DataFrame(DiffPriceSer)
    Pricedf['Date'] = Pricedf.index
    Pricedf.index = range(len(Pricedf))
    DateVolDitc = dict()
    for ind in range(len(DiffPriceSer) - 1, 0, -1):
        pricelist = []
        try:
            for j in range(ind, max(ind - 35, 0), -1):
                if (pd.Timestamp(DiffPriceSer.index[ind]) - pd.Timestamp(
                        Pricedf.loc[j, 'Date'])).days < 30:
                    pricelist.append(Pricedf.loc[j, 'UnderlyingScrtClose'])
            PriceSum = sum(np.power(pricelist, 2))
            Vol = sqrt(250 / len(pricelist) * PriceSum)
            DateVolDitc[DiffPriceSer.index[ind]] = Vol
        except ZeroDivisionError as e:
            print(ind)
            print(pricelist)
            raise e
    rvdf = pd.DataFrame([DateVolDitc.keys(), DateVolDitc.values()]).T
    rvdf.columns = ['Date', 'RealVol']
    return rvdf

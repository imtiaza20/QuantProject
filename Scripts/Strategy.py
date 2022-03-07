# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:40:05 2022

@author: lfsil
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
pd.set_option('use_inf_as_na', True)
import sys
import os
# os.chdir(r'C:\Users\Imtiaz\Desktop\Test_Run')
# path = r'C:\Users\Imtiaz\Desktop\Test_Run\Data'
# fileDir = os.path.dirname(os.path.realpath('__file__')) # Current file directory
from Preprocessing import *
from LassoRegularization import *
from RandomForestForecast import *
from MeanVariance import *


def RunStrategy(data, date_range, window, paramsLasso, paramsRF):
    output = {}
    ret_portfolio = []

    for t in date_range:
        # Get the database
        xlasso = GetLassoTest(data.loc[:t, :].shift(), window)
        ylasso = np.ones(len(xlasso))
        # Run Lasso regularization
        lasso_reg = LassoEstimation(xlasso, ylasso, paramsLasso['alpha'])
        keepAssets = (abs(lasso_reg.coef_) > 1e-2)
        keepAssets = xlasso.columns[keepAssets]
        # Running Random Forest regression
        dataRF = data[data.permno.isin(keepAssets)].loc[:t, :]
        xRF, yRF = dataRF.loc[:t, :].drop('Returns', axis=1).shift().fillna(0), \
            dataRF.loc[:t, 'Returns'].shift().fillna(0)
        xRFtest = dataRF.loc[t, :].drop('Returns', axis=1).fillna(0)
        rf_regr = RandomForestEstimation(xRF, yRF, paramsRF)
        mu = RandomForestPredict(rf_regr, xRFtest)
        cov = xlasso[keepAssets].cov()
        # Running the mean variance estimation
        wstar = MVWeights(xlasso[keepAssets].values, mu, cov)
        output[t] = [keepAssets, wstar]
        ret_portfolio.append(xlasso[keepAssets].loc[t, :] @ wstar)
    
    return output, ret_portfolio

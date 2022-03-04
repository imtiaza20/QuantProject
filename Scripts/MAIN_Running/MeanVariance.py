# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:10:52 2022

@author: lfsil
"""

import numpy as np

from scipy.optimize import minimize

def TangencyWeights(mu, cov):
    inv_cov = np.linalg.inv(cov)
    wtan = inv_cov @ mu / (np.ones(len(mu)) @ inv_cov @ mu)
    return wtan

def DeltaMV(mu_target, mu, cov, wtan):
    inv_cov = np.linalg.inv(cov)
    return mu_target * (np.ones(len(mu)) @ inv_cov @ mu) / (mu @ inv_cov @ mu)

def calmarRatio(returns):
    cumlative = np.cumprod(returns+1)
    mdd = abs(min(cumlative / np.maximum.accumulate(cumlative) - 1))
    return returns.mean() / mdd

def MVWeights(returns, mu, cov):
    wtan = TangencyWeights(mu, cov)
    def objective(mu_target, mu, cov, wtan):
        ret = DeltaMV(mu_target, mu, cov, wtan) * returns @ wtan 
        return calmarRatio(ret)
    bnds = [(0.01, 0.05)]
    mu0 = round(np.mean(bnds), 3)
    mu_t = minimize(objective, mu0, bounds = bnds, args=(mu, cov, wtan))['x']
    w = DeltaMV(mu_t, mu, cov, wtan)
    wf = w * wtan
    return wf

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:17:13 2022

@author: lfsil
"""

from sklearn.linear_model import Lasso
from functools import partial
import optuna
from optuna.trial import Trial
optuna.logging.set_verbosity(optuna.logging.FATAL)
import warnings
warnings.filterwarnings("ignore")

def LassoEstimation(x, y, alpha):
    lasso_reg = Lasso(alpha, fit_intercept=False).fit(x, y)
    
    return lasso_reg

def calmarRatio(returns):
    cumlative = (returns+1).cumprod()
    mdd = abs(min(cumlative / cumlative.cummax() - 1))

    return returns.mean() / mdd

def ObjectiveCalmar(trial:Trial, xtrain=None, xvalid = None, y=None):
    lasso_penalty = trial.suggest_float('alpha', 1e-8, 1e-5)
    lasso_reg = LassoEstimation(xtrain, y, lasso_penalty)
    weights = lasso_reg.coef_
    ret = xvalid @ weights
    
    return calmarRatio(ret)

def OptimalPenalty(objective, xtrain, xvalid, y):
    study = optuna.create_study(direction = "maximize")
    study.optimize(partial(ObjectiveCalmar, xtrain = xtrain, xvalid=xvalid, \
                           y=y), n_trials = 200, n_jobs = -1)
    params = study.best_params

    return params    


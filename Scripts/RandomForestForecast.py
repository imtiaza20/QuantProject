# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 08:27:01 2022

@author: lfsil
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from functools import partial
import optuna
from optuna.trial import Trial
optuna.logging.set_verbosity(optuna.logging.FATAL)
import warnings
warnings.filterwarnings("ignore")


def RandomForestEstimation(xtrain, ytrain, params):
    rf_regr = RandomForestRegressor(**params)
    rf_regr.fit(xtrain, ytrain)
    return rf_regr

def RandomForestPredict(rf_regr, xtest):
    return rf_regr.predict(xtest)

def ObjectiveRsquared(trial:Trial, xtrain=None, xval=None, ytrain=None, yval=None):
    rf_n_estimators = trial.suggest_int('n_estimators', 10, 50, step=5)
    rf_max_depth = trial.suggest_int('max_depth', 4, 15)
    rf_max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    regr = RandomForestRegressor(max_depth = rf_max_depth, \
                                n_estimators = rf_n_estimators,\
                                max_features = rf_max_features,\
                                random_state = 0)
    regr.fit(xtrain, ytrain)
    preds = regr.predict(xval)
    
    return r2_score(yval, preds)

def OptimalHyperParameters(objective, xtrain, xval, ytrain, yval):
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(ObjectiveRsquared, xtrain=xtrain, xval=xval, \
                       ytrain=ytrain.values, yval=yval.values), n_trials = 200, \
               n_jobs = -1)
    params = study.best_params
    
    return params

    









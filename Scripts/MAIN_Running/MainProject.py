# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:47:43 2022

@author: lfsil
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
pd.set_option('use_inf_as_na', True)
import sys
import os
os.chdir(r'C:\Users\lfsil\Documents\Chicago\Regression Analysis and Quant Trading\Assignments\Project\Scripts')
path = r'C:\Users\lfsil\Documents\Chicago\Regression Analysis and Quant Trading\Assignments\Project\Scripts\Data'
from Preprocessing import *
from LassoRegularization import *
from RandomForestForecast import *
from Strategy import *
import matplotlib.pyplot as plt
%matplotlib inline

# =============================================================================
# Step 1. Choosing the parameters of the model
# =============================================================================

perc = 0.4 # Fraction of data to be used in the Random Forest model
ratio = (60, 20, 20) # Train, valid and test split
window = 100 # Minimum amount of data for an asset to be kept in the model
attributes = ['Returns', 'Capei', 'debtToEquity', 'dividendYield', \
                'priceToBook', 'priceToCashflow', 'priceToSales',\
                'returnOnAssets', 'returnOnEquity']

# =============================================================================
# Step 2. Calculating optimal hyperparameters
# =============================================================================

data = ProcessData(attributes)
train, valid, test = TrainTestSplit(data, ratio)
trainRF, validRF, testRF = CleanRandomForest(train, valid, test, perc)
xtrain, ytrain = SplitXandY(trainRF)
xval, yval = SplitXandY(validRF)
xtest, ytest = SplitXandY(testRF)
# Parameters for the Random Forest model
paramsRF = OptimalHyperParameters(ObjectiveRsquared, xtrain, xval, ytrain, yval)
# Parameters for the Lasso Regression
retTrain, retValid = CleanLasso(train, valid)
paramsLasso = OptimalPenalty(ObjectiveCalmar, retTrain, retValid, \
                             np.ones(len(retTrain)))

# =============================================================================
# Step 3. Running the model
# =============================================================================

date_range = test.index.unique()
output, ret_portfolio = RunStrategy(data, date_range, window, paramsLasso, paramsRF)
   
# =============================================================================
# Step 4. Analyzing results
# =============================================================================

cumret = np.cumprod(1 + np.array(ret_portfolio))
plt.figure()
plt.plot(cumret, label = 'Cummulative returns')
plt.legend()
plt.show()



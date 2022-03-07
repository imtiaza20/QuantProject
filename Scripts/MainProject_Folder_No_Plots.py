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
from os import makedirs, path
from errno import EEXIST
# os.chdir(r'C:\Users\Imtiaz\Desktop\Test_Run')
# path = r'C:\Users\Imtiaz\Desktop\Test_Run\Data'
# fileDir = os.path.dirname(os.path.realpath('__file__')) # Current file directory
from Preprocessing import *
from LassoRegularization import *
from RandomForestForecast import *
from Strategy import *
import matplotlib.pyplot as plt
# %matplotlib inline

## Creates a directory. equivalent to using mkdir -p on the command line
def mkdir_p(mypath):
    # This will override files in the folder if they exist AND have the same name  
    try:
        makedirs(mypath)
    except OSError as exc: # Python > 2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: 
            raise


if __name__ == '__main__':
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
    # plt.figure()
    # plt.plot(cumret, label = 'Cummulative returns')
    # plt.legend()
    # plt.show()

    # =============================================================================
    # Step 5. Store the variables you want
    # =============================================================================
    
    ## Create new directory to automatically save MSD data
    n = int(sys.argv[1])
    output_dir1 = "Strategy"
    mkdir_p(output_dir1)

    ## Make file names and store text file into designated directory
    file_name1 = "cumret_{}.csv".format(n)
    file_name2 = "ret_portfolio_{}.csv".format(n)
    file_path1 = os.path.join(output_dir1, file_name1) # Stores txt file to the designated directory
    file_path2 = os.path.join(output_dir1, file_name2) 
    
    ## Save data
    np.savetxt(file_path1, cumret, delimiter = ',') 
    np.savetxt(file_path2, ret_portfolio, delimiter = ',')
    



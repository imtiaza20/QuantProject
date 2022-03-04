# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:50:14 2022

@author: lfsil
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

path = r"C:\Users\lfsil\Documents\Chicago\Regression Analysis and Quant Trading\Assignments\Project\Scripts\Data"

def ProcessData(attributes):
        
    for i, attr in enumerate(attributes):
        colnames = ['date', 'permno', attr]
        if i == 0:
            df = pd.read_csv(path + '/' + attr + '.txt', sep=',').\
                stack(dropna=False).reset_index()
            df.columns = colnames
        else:
            df_new = pd.read_csv(path + '/' + attr + '.txt', sep=',').\
                stack(dropna = False).reset_index()
            df_new.columns = colnames 
            df = df.merge(df_new, how = "left", on = ["date", "permno"])
    df = df.set_index('date').sort_index()

    return df
    
def TrainTestSplit(df, ratio):
    dates = df.index.unique().values
    rtrain, rvalid, _ = ratio[0], ratio[1], ratio[2]
    ntrain = int(len(dates) * rtrain / 100)
    nvalid = int(len(dates) * (rtrain + rvalid) / 100)
    start_train, end_train = dates[0], dates[ntrain]
    start_valid, end_valid = dates[ntrain + 1], dates[nvalid]
    start_test, end_test = dates[nvalid+1], dates[-1]
    train, valid, test = df.loc[:end_train, :], \
        df.loc[start_valid:end_valid, :], df.loc[start_test:, :]
        
    return train, valid, test

def CleanRandomForest(dftrain, dfvalid, dftest, perc):
    # Condition: keep stocks with at least some percentage of possible 
    # observations in each feature in each data set
    nfeatures = dftrain.shape[1] - 1
    nobsTrain, nobsValid, nobsTest = len(dftrain.index.unique()),\
        len(dfvalid.index.unique()), len(dftest.index.unique())

    conditionTrain = ((dftrain.groupby('permno').count() > int(nobsTrain * perc)).\
        sum(axis=1) == nfeatures)
    permnoTrain = conditionTrain[conditionTrain.values].index.values
    conditionValid = ((dfvalid.groupby('permno').count() > int(nobsValid * perc)).\
        sum(axis=1) == nfeatures)
    permnoValid = conditionValid[conditionValid.values].index.values
    conditionTest = ((dftest.groupby('permno').count() > int(nobsTest * perc)).\
        sum(axis=1) == nfeatures)
    permnoTest = conditionTest[conditionTest.values].index.values
    trainRF, validRF, testRF = \
        dftrain[dftrain.permno.isin(permnoTrain).values].fillna(0),\
        dfvalid[dfvalid.permno.isin(permnoValid).values].fillna(0),\
        dftest[dftest.permno.isin(permnoTest).values].fillna(0)
        
    return trainRF, validRF, testRF
        
def SplitXandY(data):
    x = data.drop('Returns', axis=1)
    y = data.loc[:, 'Returns']
    return x, y
        
def CleanLasso(xtrain, xvalid):
    retTrain = pd.pivot_table(data=xtrain.reset_index(), values = 'Returns', \
                         index = 'date', columns = 'permno')
    retValid = pd.pivot_table(data=xvalid.reset_index(), values = 'Returns', \
                         index = 'date', columns = 'permno')
    permnoKeep = retTrain.columns[retTrain.columns.isin(retValid.columns)]
    retTrain = retTrain.loc[:, permnoKeep]
    retValid = retValid.loc[:, permnoKeep]
    imputerTrain = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputerTrain.fit(retTrain)
    imputerValid = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputerValid.fit(retValid)
    retTrain = pd.DataFrame(imputerTrain.transform(retTrain), \
                            index=retTrain.index, columns=retTrain.columns)
    retValid = pd.DataFrame(imputerValid.transform(retValid), \
                            index = retValid.index, columns = retValid.columns)

    return retTrain, retValid

def GetLassoTest(df, window):
    dflasso = pd.pivot_table(data=df.reset_index(), values = 'Returns', \
                             index = 'date', columns = 'permno')
    dflasso.iloc[-window:, :]
    condition = (~dflasso.iloc[-window:, :].isna()).sum(axis=0) >= window
    permnoKeep = condition[condition.values].index.values
    dflasso = dflasso.loc[:, permnoKeep]
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(dflasso)
    
    return pd.DataFrame(imputer.transform(dflasso), index=dflasso.index, \
                        columns=dflasso.columns)


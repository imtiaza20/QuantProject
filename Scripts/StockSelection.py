import Library as lib
import pandas as pd
import numpy as np
import optuna as opt
from optuna.trial import Trial
from functools import partial

file = pd.read_csv("Returns.txt", sep=",")
returns = pd.DataFrame(file.iloc[0:60,:].dropna(1))

def Huber(epsilon, c):
	if abs(epsilon) <= c:
		output = (1/2) * (epsilon ** 2)
	else:
		output = c * abs(epsilon) - (1/2) * (c ** 2)
	return output

def HuberLasso(data, betas, Lambda, c):
	betas.index = data.columns
	y = pd.Series(np.ones(data.shape[0]))
	epsilon = data.dot(betas).squeeze()
	epsilon = epsilon.apply(lambda x: Huber(x, c))
	obj = epsilon.sum() / data.shape[0] + Lambda * abs(betas[1:]).sum()
	return obj

def HuberL0(data, betas, Lambda, c):
	betas.index = data.columns
	y = pd.Series(np.ones(data.shape[0]))
	epsilon = data.dot(betas).squeeze()
	epsilon = epsilon.apply(lambda x: Huber(x, c))
	obj = epsilon.sum() / data.shape[0] + Lambda * (betas[1:] != 0).sum()
	return obj

def HuberLassoBetas(trial:Trial, data = None, Lambda = None, c = None):
	names = data.columns
	betas = []
	for i in range(len(names)):
		betas.append(trial.suggest_float(names[i], -1, 1))
	betas = pd.Series(betas)
	return(HuberLasso(data, betas, Lambda, c))


study = opt.create_study(direction = "minimize")
study.optimize(partial(HuberLassoBetas, data = returns, Lambda = .001, c = 1), n_trials = 200, n_jobs = -1)

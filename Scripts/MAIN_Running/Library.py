import math
import numpy as np
import pandas as pd

def informationRatio(returns, benchmark):
	beta = returns.cov(benchmark) / benchmark.var()
	alpha = returns.mean() - beta * benchmark.mean()
	epsilon = returns - (alpha + beta * benchmark)
	return alpha / epsilon.std() * math.sqrt(12)

def sharpeRatio(returns):
	return returns.mean() / returns.std() * math.sqrt(12)

def calmarRatio(returns):
    cumlative = (returns+1).cumprod()
    mdd = abs(min(cumlative / cumlative.cummax() - 1))
    return returns.mean() / mdd

def informationCalmar(returns, benchmark):
	beta = returns.cov(benchmark) / benchmark.var()
	alpha = returns.mean() - beta * benchmark.mean()
	epsilon = returns - (alpha + beta * benchmark)
	cumlative = (epsilon+1).cumprod()
	mdd = abs(min(cumlative / cumlative.cummax() - 1))
	return alpha / mdd

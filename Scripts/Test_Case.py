import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# reading the CSV file
ret_portfolio = pd.read_csv('ret_portfolio.csv')
cumret = np.cumprod(1 + np.array(ret_portfolio))
print(type(cumret))
plt.figure()
plt.plot(cumret, label = 'Cummulative returns')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:06:10 2020

@author: jyoth
"""

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Create the regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
regressor.fit(X, y)

# Predict the salary of level 6.5
y_pred = regressor.predict((6.5))

# Visualize the Random Forest Regresson in high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, marker = '*', color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.title('Truth or bluff of employee')
plt.xlabel('Level or Position')
plt.ylabel('Salary')
plt.show()
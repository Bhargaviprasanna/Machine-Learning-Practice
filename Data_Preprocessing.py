# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:55:52 2020

@author: jyoth
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', axis = 0, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#enconding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0]) 
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)
#making ml model to not to calculate countries based on age
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Scaling the data (-1 to 1 range)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Visualising using swarm plot 
import seaborn as sns
sns.swarmplot(x='Country', y='Salary', data=dataset, hue='Purchased')
sns.swarmplot(x='Purchased', y='Salary', data=dataset, hue='Country')
plt.show()
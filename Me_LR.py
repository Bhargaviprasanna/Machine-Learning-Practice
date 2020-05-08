# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:34:20 2020

@author: jyoth
"""

# Importing the datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 4].values

# Encoding for male or female in X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Label_X = LabelEncoder()
X[:, 0] = Label_X.fit_transform(X[:, 0])
one_x = OneHotEncoder(categorical_features= [0])
X = one_x.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling categorical variables should not be encoded
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 2:] = sc_X.fit_transform(X_train[:, 2:])
X_test [:, 2:]= sc_X.transform(X_test[:, 2:])

# Creating classifer object
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting y
y_pred = classifier.predict(X_test)

# Check how many predicitons are true, false using function confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 

"""X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))
Z = np.array([X1.ravel(), X2.ravel()]).T
plt.pcolor(classifier.predict(Z), cmap = 'blues')
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.colorbar()
plt.show() """
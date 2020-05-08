# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:38:50 2020

@author: jyoth
"""
#It is a binary classification

# Importing the datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling categorical variables should not be encoded
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating classifer object
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Prediciting 
y_pred = classifier.predict(X_test)
# Check how many predicitons are true, false using function confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Dependent variable in the dataset
import seaborn as sns
sns.countplot(x = 'Purchased', data = dataset, palette = 'hls')

# Visualising the 'Age' and 'Estimated salary' as scatterplot in Whole set
purchased = dataset.iloc[y == 1] #Filtering purchased data
not_purchased = dataset.iloc[y == 0] # Filtering Unpurchased data
plt.scatter(purchased.iloc[:, 2], purchased.iloc[:, 3], s=30, label = 'Purchased', color = 'red')
plt.scatter(not_purchased.iloc[:, 2], not_purchased.iloc[:, 3], s=30, label = 'Not_Purchased', color = 'blue')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the 'Age' and 'Estimated salary' as scatterplot in Training set
purchased = X_train[y_train == 1]
not_purchased = X_train[y_train == 0]
plt.scatter(purchased[:, 0], purchased[:, 1], label = 'Purchased', color = 'red')
plt.scatter(not_purchased[:, 0], not_purchased[:, 1], label = 'Not_Purchased', color = 'blue')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the 'Age' and 'Estimated salary' as scatterplot in Test set
purchased = X_test[y_test == 1]
not_purchased = X_test[y_test == 0]
plt.subplot(2, 2, 1)
plt.title('Test_set')
plt.scatter(purchased[:, 0], purchased[:, 1], label = 'Purchased', color = 'red')
plt.scatter(not_purchased[:, 0], not_purchased[:, 1], label = 'Not_Purchased', color = 'blue')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
#plt.legend()
plt.subplot(2, 2, 2)
plt.title('Predicted')
purchased = X_test[y_pred == 1]
not_purchased = X_test[y_pred == 0]
plt.scatter(purchased[:, 0], purchased[:, 1], label = 'Purchased', color = 'green')
plt.scatter(not_purchased[:, 0], not_purchased[:, 1], label = 'Not_Purchased', color = 'orange')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
#plt.legend()
plt.tight_layout()
plt.show()

#Visualising the Logistic Regression classifier 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'gray')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# final Visualisation of Classifier with prediction results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'gray')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title('Predicted')
purchased = X_test[y_pred == 1]
not_purchased = X_test[y_pred == 0]
plt.scatter(purchased[:, 0], purchased[:, 1], label = 'Purchased', color = 'green')
plt.scatter(not_purchased[:, 0], not_purchased[:, 1], label = 'Not_Purchased', color = 'orange')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(fontsize = 'x-small')
plt.show()


# Polynomial Regression by me
""" An Employer said his prev salary is 160k dollars, so HR team of new company checking whether he is telling the truth or not he is an regional manager position in last company """
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
import seaborn as sns
plt.scatter(X, y, color = 'red')
sns.lmplot(x='Level', y='Salary', data=dataset)
# Visualize the positions as subplot in a row 
sns.lmplot(x='Level', y='Salary', data=dataset, row='Position')
# Visualize the positions as subplot in a column 
sns.lmplot(x='Level', y='Salary', data=dataset, col='Position')
# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree = 2)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the Linear model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Salaries)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Poly model
plt.scatter(X, y, color = 'red')
sns.regplot(x = 'Level', y = 'Salary', data = dataset, color = 'yellow', order = 3)
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Truth or Bluff (Salaries)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

# Visualising the Poly model with x-axis step by 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.subplot(2, 2, 1)
plt.scatter(X, y, color = 'orange')
#sns.regplot(x = 'Level', y = 'Salary', data = dataset, color = 'yellow', order = 3)
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth or Bluff (Salaries)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.subplot(2, 2, 2)
plt.scatter(X, y, color = 'blue')
#sns.regplot(x = X_grid, y=poly_reg.fit_transform(X_grid), color = 'red', order = 3)
plt.tight_layout()
plt.title('Truth or Bluff (Salaries)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising using Seaborn
import seaborn as sns
sns.regplot(x = 'Level', y = 'Salary', data = dataset, color = 'blue', order = 4)
#sns.regplot(x='weight', y='mpg', data=auto, , scatter=None, label='First Order')

# Predicting the lin model salaries
y_lin_pred = lin_reg.predict(X)
lin_reg.predict(6.5)
# Prediciting the poly model salaries
y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))
lin_reg2.predict(poly_reg.fit_transform(6.5)) #for 6.5 level 

""" The predicted sal is 159k so near to 160k answer is employee is Telling the Truth"""

# Constructing a strip plot
sns.stripplot(x='Salary', y='Position', data=dataset, size=6, jitter = True, orient = 'h')
plt.show()
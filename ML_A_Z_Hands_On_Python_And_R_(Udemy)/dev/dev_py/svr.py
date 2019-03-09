# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values.astype(float)
y = dataset.iloc[:, 2].values.astype(float)
# astype float is to take care of DataConversion Error

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = X.reshape(-1,1)
y = y.reshape(-1,1)
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
# Note: 6.5 is float, [6.6] is vector, 
#       np.array([6.5]) is array or mtrx

#y_pred = regressor.predict(6.5)  # this gives deprecation error
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

y_pred = regressor.predict(sc_X.transform(np.array([6.5])))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred) # ans = [ 170370.0204065]

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Warnings
"""
1. warnings.warn(msg, _DataConversionWarning)
2. warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning) 
   passing 1d arrays as data in sk.preprocessing
3. warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
4. warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
5. DeprecationWarning)
"""

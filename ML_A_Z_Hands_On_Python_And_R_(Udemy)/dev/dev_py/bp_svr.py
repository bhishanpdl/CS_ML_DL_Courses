# SVR

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# avoid DataConversionError
X = X.astype(float)
y = y.astype(float)


## Attempt to avoid DeprecationWarning for sk.prepossing
#X = X.reshape(-1,1)                  # attempt 1
#X = np.array(X).reshape((len(X), 1)) # attempt 2
#X = np.array([X])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([6.5])))
y_pred = sc_y.inverse_transform(y_pred)


"""
As part of sklearn.pipeline stages' uniform interfaces, as a rule of thumb:

when you see X, it should be an np.array with two dimensions
when you see y, it should be an np.array with a single dimension.
"""

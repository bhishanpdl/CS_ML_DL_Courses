#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author      : Bhishan Poudel; Physics PhD Student, Ohio University
# Date        : May 09, 2017
# Last update : 
#

# Imports
import pydoc
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Add a column of 50 ones at the beginning of X
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
# Initialize optimal X as original matrix, we will backprop later.
X_opt = X[:,[0,1,2,3,4,5]]  
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # ordinary least sq
print(regressor_OLS.summary())

# Look for highest Probability (lowest significance) and remove this
# remove if it is above 5%.
X_opt = X[:,[0,1,3,4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # ordinary least sq
print(regressor_OLS.summary())
# remove if it is above 5%.
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # ordinary least sq
print(regressor_OLS.summary())
# remove if it is above 5%.
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # ordinary least sq
print(regressor_OLS.summary())

# some helps
#print(pydoc.render_doc(np.append, "Help on %s"))
#print(pydoc.render_doc(sm.OLS, "Help on %s"))
#print(help(sm.OLS))

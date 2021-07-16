#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author      : Bhishan Poudel; Physics PhD Student, Ohio University
# Date        : May 09, 2017
# Last update : July 1, 2017
#
#
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as DF

# Import dataset
dataset = pd.read_csv('data/Salary_Data.csv')
# print(DF(dataset))

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# print(DF(X))


# Split data into train/test
from sklearn.model_selection import train_test_split

# ?train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Fitting the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Visualize
plt.scatter(X_train, y_train, c='r')
plt.plot(X_train, regressor.predict(X_train), c='b')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Visualize Test set
# plt.scatter(X_test,y_test,c='r')
# plt.plot(X_test, regressor.predict(X_test), c='b')
# plt.title('Salary vs Experience (Test Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
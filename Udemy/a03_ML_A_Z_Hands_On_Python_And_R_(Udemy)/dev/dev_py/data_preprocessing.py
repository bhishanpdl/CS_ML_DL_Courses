#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author      : Bhishan Poudel; Physics PhD Student, Ohio University
# Date        : Jun 07, 2017 Wed
# Last update :
#
# Imports
# Data Processing
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')
X  = dataset.iloc[:,:-1].values
X2 = dataset.iloc[:,:-1].values
y  = dataset.iloc[:,3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer


# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X[:, 1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

# one liner
X[:, 1:3] = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(X[:,1:3]).transform(X[:,1:3])

# PART 2
# Categorical values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])


# Create dummy encoder since France=0 is not smaller than Spain=1 or so on
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

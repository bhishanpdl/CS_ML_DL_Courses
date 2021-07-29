# Importing the libraries
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler

# Setting up data string to be read in as a .csv
data = StringIO("""Position,Level,Salary
Business Analyst,1,45000
Junior Consultant,2,50000
Senior Consultant,3,60000
Manager,4,80000
Country Manager,5,110000
Region Manager,6,150000
Partner,7,200000
Senior Partner,8,300000
C-level,9,500000
CEO,10,1000000""")

dataset = pd.read_csv(data)

# Importing the dataset
#dataset = pd.read_csv('Position_Salaries.csv')

# Deprecation warnings call for reshaping of single feature arrays with reshape(-1,1)
X = dataset.iloc[:, 1:2].values.reshape(-1,1)
y = dataset.iloc[:, 2].values.reshape(-1,1)

# avoid DataConversionError
X = X.astype(float)
y = y.astype(float)

#sc_X = StandardScaler()
#sc_y = StandardScaler()
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)

X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

# One of the warnings called for ravel()
regressor.fit(X_scaled, y_scaled.ravel())

# Predicting a new result
# The warnings called for single samples to reshaped with reshape(1,-1)
X_new = np.array([6.5]).reshape(1,-1)
X_new_scaled = X_scaler.transform(X_new)
y_pred = regressor.predict(X_new_scaled)
y_pred = y_scaler.inverse_transform(y_pred)
print(y_pred)

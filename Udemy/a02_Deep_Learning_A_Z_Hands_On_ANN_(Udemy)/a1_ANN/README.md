Table of Contents
=================
   * [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
   * [Building Classifier](#building-classifier)
   * [Dropout](#dropout)
   * [Parameter tuning](#parameter-tuning)

# Artificial Neural Network (ANN)
```python
# Load data
#---------------------------------------------------------------------------------
import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/Churn_Modelling.csv')
X = df.iloc[:,3:-1].values
y = df.iloc[:, -1].values

# Preprocessing
#---------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encode geography country names
le_geo = LabelEncoder()
X[:, 1] = le_geo.fit_transform(X[:, 1])

# encode gender male female
le_gen = LabelEncoder()
X[:, 2] = le_gen.fit_transform(X[:, 2])

ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:] # do not take first dummy variable

# Data Splitting
#---------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Data Standardizing
#---------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Make ANN
#---------------------------------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Build classifier
clf = Sequential()
clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))
clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# fitting
clf.fit(X_train,y_train, batch_size=10, nb_epoch=100)

# prediction
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
acc = (cm[0,0] + cm[1,1]) / np.sum(cm.flatten())
print(acc)

# Accuracy prediction on test data
#---------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
acc = (cm[0,0] + cm[1,1]) / np.sum(cm.flatten())
```

# Building Classifier
```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    clf = Sequential()
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    clf.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return clf

clf = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
acc = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, n_jobs=-1)

mean = acc.mean()
std = acc.std()
```

# Dropout
```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout

# Build classifier
clf = Sequential()

clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))
clf.add(Dropout(0.1))

clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
clf.add(Dropout(0.1))

clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# fitting
clf.fit(X_train,y_train, batch_size=10, epochs=100)

# prediction
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
acc = (cm[0,0] + cm[1,1]) / np.sum(cm.flatten())
print(acc)
```

# Parameter tuning
```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    clf = Sequential()
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    clf.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    return clf

clf = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32],
             'epochs': [100,500],
             'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=clf,
                          param_grid = parameters,
                          scoring='accuracy',
                          cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_acc = grid_search.best_score_

print(best_parameters, best_acc)
```

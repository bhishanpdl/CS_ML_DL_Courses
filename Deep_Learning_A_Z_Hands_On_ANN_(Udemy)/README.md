Table of Contents
=================
   * [Course Introduction](#course-introduction)
   * [Loading data](#loading-data)
   * [Missing data](#missing-data)
   * [Encoding categorical variables](#encoding-categorical-variables)
   * [Label Encoder vs. One Hot Encoder](#label-encoder-vs-one-hot-encoder)
   * [Splitting data](#splitting-data)
   * [Feature scaling](#feature-scaling)
   * [Fitting classification](#fitting-classification)
   * [Predicting test set](#predicting-test-set)
   * [Confusion matrix](#confusion-matrix)

# Course Introduction
> Course Name: Deep Learning A-Z™: Hands-On Artificial Neural Networks  
> Course website: https://www.udemy.com/deeplearning/learn/v4/content  
> Dataset: https://www.superdatascience.com/deep-learning/  


# Loading data
```python
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
```

# Missing data
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

# Encoding categorical variables
```python
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```

# Label Encoder vs. One Hot Encoder
```python
Country LabelEncode   OHE
France  0             0 1
Spain   1             1 0
# We should drop first column of ohe, to avoid dummy varible problem.
```

# Splitting data
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

# Feature scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

# Fitting classification
```python
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
```

# Predicting test set
```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # for binary classification
```

# Confusion matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```


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

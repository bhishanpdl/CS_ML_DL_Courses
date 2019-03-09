import numpy as np 
from sklearn import svm 
import random
A = np.array([[random.randint(0, 20) for i in range(2)] for i in range(10)]) 
lab = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

clf = svm.SVC(kernel='linear', C=1.0) 
clf.fit(A, lab)

print (clf.predict([[1,9]]))


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

x_train = x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)

x_train.shape

x_test.shape

x_train=x_train/255
x_test=x_test/255

"""## Linear Model"""

model = SVC(kernel='linear',random_state=22) #roll no 22
model.fit(x_train,y_train)
pred = model.predict(x_test)

print(pred)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print("Precision Score : ",metrics.precision_score(y_test, pred, pos_label='positive' ,average='micro'))
print("Recall Score : ",metrics.recall_score(y_test, pred, pos_label='positive',average='micro'))

"""## Polynomial Model"""

model1 = SVC(kernel='poly',degree=3,gamma='scale',random_state=22) #roll no 22
model1.fit(x_train,y_train)
pred1 = model1.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, pred1))
print("Precision Score : ",metrics.precision_score(y_test, pred1, pos_label='positive' ,average='micro'))
print("Recall Score : ",metrics.recall_score(y_test, pred1, pos_label='positive',average='micro'))

"""## RBF Model"""

from sklearn.svm import SVC
model2 = SVC(kernel='rbf',gamma='scale',random_state=22)
model2.fit(x_train,y_train)
pred2 = model2.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, pred2))
print("Precision Score : ",metrics.precision_score(y_test, pred2, pos_label='positive' ,average='micro'))
print("Recall Score : ",metrics.recall_score(y_test, pred2, pos_label='positive',average='micro'))
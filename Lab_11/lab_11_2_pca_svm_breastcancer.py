# %matplotlib inline
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import svm,decomposition
import matplotlib.pyplot as plt
import numpy as np 

dataset = datasets.load_breast_cancer()
dataset.target_names

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.data ,dataset.target, random_state=22)

print(X_train.shape, X_test.shape)
# plotting scatters
plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target, s=25,cmap='spring');
plt.show()

pca = decomposition.PCA(n_components=20, whiten=True, random_state=22)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

clf = svm.SVC(C=2., gamma=0.005, random_state=22)
clf.fit(X_train_pca, y_train)

from sklearn import metrics
y_pred = clf.predict(X_test_pca)

print(metrics.classification_report(y_test, y_pred))

from sklearn.pipeline import Pipeline
clf = Pipeline([('pca', decomposition.PCA(n_components=2, whiten=True)), ('svm', svm.LinearSVC(C=2.0))])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(metrics.confusion_matrix(y_pred, y_test))
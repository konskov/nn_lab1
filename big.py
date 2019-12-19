# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:27:48 2019

@author: skovola
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier #dummy
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier #mlp
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB #gnb
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


data_filepath = r'C:\Users\skovola\Downloads\dermatology.data' #os.path.join(os.getcwd(), 'C:\Users\skovola\Downloads\dermatology.data')

# Read data #comment
data = pd.read_csv(data_filepath, header=None, na_values=['?']) #comment
#data.to_pickle('data') #comment
#data = pd.read_pickle('data')
# Split labels and features
X, y = data.iloc[:, :-1], data.iloc[:, -1]
labels = y.unique()
label_frequencies = y.value_counts(normalize=True)
print(len(label_frequencies))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42069)
imr = SimpleImputer(missing_values=np.NaN, strategy='mean')
imr = imr.fit(X_train)
X_train = imr.transform(X_train)
#print('X_train size: %s' %(X_train.shape,))
X_test = imr.transform(X_test)

#knn
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_preds = knn.predict(X_test)
#print(classification_report(y_test, knn_preds))

# Dummy classifier
dummy = DummyClassifier(strategy="stratified")
dummy.fit(X_train, y_train)
dc_y_pred = dummy.predict(X_test)
#print(classification_report(y_test,dc_y_pred))

# Gaussian naive Bayes
gnb = GaussianNB()
# κάνουμε εκπαίδευση (fit) δηλαδή ουσιαστικά υπολογίζουμε μέση τιμή και διακύμανση για όλα τα χαρακτηριστικά και κλάσεις στο training set
gnb.fit(X_train, y_train)
gnb_preds = gnb.predict(X_test)
#print(classification_report(y_test,gnb_preds))



# Multi layer perceptron
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,), random_state=1)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
#print(classification_report(y_test, mlp_preds))

# αρχικοποιούμε τους εκτιμητές (μετασχηματιστές και ταξινομητή) χωρείς παραμέτρους
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
knn = KNeighborsClassifier(n_jobs=-1) # η παράμετρος n_jobs = 1 χρησιμοποιεί όλους τους πυρήνες του υπολογιστή
#consruct pipe
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('kNN', knn)])
pipe.fit(X_train,y_train)
preds = pipe.predict(X_test)
print(classification_report(y_test, preds))

train_variance = X_train.var(axis=0)
print(np.sort(train_variance))
print(np.mean(train_variance))
print(np.min(train_variance))
print(np.max(train_variance))

vthreshold = [0, 0.3, 0.6, 0.9] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε
n_components = [2, 4, 6, 8, 9, 10] #PCA Parameter
k = [1, 3, 5, 7, 9, 11] # η υπερπαράμετρος του ταξινομητή KNN
layers = [2,3,4,5] # η υπερπαράμετρος του ταξινομητή MLP
#class_probs = [None,[label_frequencies[i] for i in range(len(label_frequencies))]] 

pipe_dummy = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), 
                       ('pca', pca), ('dummy', dummy)], memory = 'tmp')
estimator_gnb = GridSearchCV(pipe_dummy, dict(selector__threshold=vthreshold, 
                                    pca__n_components=n_components), 
cv=5, scoring='f1_macro', n_jobs=-1)

pipe_gnb = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), 
                       ('pca', pca), ('gnb', gnb)], memory = 'tmp')
estimator_gnb = GridSearchCV(pipe_gnb, dict(selector__threshold=vthreshold, 
                                    pca__n_components=n_components), 
cv=5, scoring='f1_macro', n_jobs=-1)

pipe_knn = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), 
                       ('pca', pca), ('kNN', knn)], memory = 'tmp')
estimator_knn = GridSearchCV(pipe_knn, dict(selector__threshold=vthreshold, 
                                    pca__n_components=n_components, kNN__n_neighbors=k), 
cv=5, scoring='f1_macro', n_jobs=-1)

pipe_mlp = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), 
                       ('pca', pca), ('mlp', mlp)], memory = 'tmp')
estimator_mlp = GridSearchCV(pipe_mlp, dict(selector__threshold=vthreshold, 
                                    pca__n_components=n_components, mlp__hidden_layer_sizes=layers),
cv=5, scoring='f1_macro', n_jobs=-1)

for estimator_name in ['estimator_knn','estimator_mlp','estimator_gnb']:
    estimator = eval(estimator_name)
    print('for estimator %s' %(estimator_name))    
    start_time = time.time()
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
    print(classification_report(y_test, preds))
    
    print(estimator.best_estimator_)
    print()
    print(estimator.best_params_)
    


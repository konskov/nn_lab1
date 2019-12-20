# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:56:18 2019

@author: skovola
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
#from gridsearch import grid_search_cv

data_filepath = r'C:\Users\skovola\Documents\Σχολη\9ο εξαμηνο\nnlab1\Nomao\Nomao.data'

# Read data #comment
data = pd.read_csv(data_filepath, header=None, na_values=['?']) #comment
print(data[11])

# all samples have some missing values

# Split labels and features
X, y = data.iloc[:, :-1], data.iloc[:, -1]

labels = y.unique()
label_frequencies = y.value_counts(normalize=True)
instability_metric = label_frequencies.max() / label_frequencies.min()

print(
    f'Number of labels: {len(labels)}\n'
    f'Label frequencies: \n{label_frequencies.to_string()}\n'
    f'Is the sample unstable? {instability_metric > 1.5}\n'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42069)

#print(type(X_train)) #pandas data frame

num_X_train = X_train.select_dtypes(include='number')
num_X_test = X_test.select_dtypes(include='number')
#print(data[7])
#### fill in missing values ####
imr = SimpleImputer(missing_values=np.NaN, strategy='mean')
imr = imr.fit(num_X_train)
X_train = imr.transform(num_X_train)
#print('X_train size: %s' %(X_train.shape,))
X_test = imr.transform(num_X_test)
# now fix remaining data
# do it separately because now the replacement strategy must be 
# most frequent for categorical values

imr = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imr = imr.fit(X_train)
X_train = imr.transform(X_train)
#print('X_train size: %s' %(X_train.shape,))
X_test = imr.transform(X_test)
nsm_mapping = {'n': 0, 's': 1, 'm': 2}
data = data.applymap(lambda s: nsm_mapping.get(s) if s in nsm_mapping else s)

#done preprocessing
train_variance = X_train.var(axis=0)
print(np.sort(train_variance))
print(np.min(train_variance))
print(np.max(train_variance)) #0.18
print(np.mean(train_variance)) #0.0405



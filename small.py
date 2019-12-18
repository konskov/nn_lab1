import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

data_filepath = os.path.join(os.getcwd(), 'dermatology.data')

# Read data
# data = pd.read_csv(data_filepath, header=None, na_values=['?'])
# data.to_pickle('data')
data = pd.read_pickle('data')

# ########### Step B ############
# Count missing values
temp = data.isna().sum(axis=1)
nan_samples = temp.astype(bool).sum()
percentage = nan_samples / data.shape[0]

print(
    f'Number of samples with missing values: {nan_samples}\n'
    f'Percentage of total samples: {percentage}\n'
)

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

# Fill in missing values
imr = SimpleImputer(missing_values=np.NaN, strategy='mean')
imr = imr.fit(X_train)
X_train = imr.transform(X_train)
X_test = imr.transform(X_test)

# There are no categorical features to encode

# ########### Step C ############
dc_stratified = DummyClassifier(strategy="stratified")
dc_stratified.fit(X_train, y_train)
dc_y_pred = dc_stratified.predict(X_test)
dc_conf = confusion_matrix(y_test, dc_y_pred)
dc_f1_micro = f1_score(y_test, dc_y_pred, average='micro')
dc_f1_macro = f1_score(y_test, dc_y_pred, average='macro')

print(
    'Dummy Classifier Results\n'
    f'Confusion Matrix:\n{dc_conf}\n'
    f'f1-micro average: {dc_f1_micro}\n'
    f'f1-macro average: {dc_f1_macro}\n'
)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_conf = confusion_matrix(y_test, knn_y_pred)
knn_f1_micro = f1_score(y_test, knn_y_pred, average='micro')
knn_f1_macro = f1_score(y_test, knn_y_pred, average='macro')

print(
    'k-NN Classifier Results\n'
    f'Confusion Matrix:\n{knn_conf}\n'
    f'f1-micro average: {knn_f1_micro}\n'
    f'f1-macro average: {knn_f1_macro}\n'
)

plt.figure()


plt.subplot('211')
plt.title('f1-scores for classification')
plt.ylabel('f1-micro average')
plt.bar([0, 1], [dc_f1_micro, knn_f1_micro], tick_label=['Stratified Dummy', 'k Nearest Neighbors'])

plt.subplot('212')
plt.ylabel('f1-macro average')
plt.bar([0, 1], [dc_f1_macro, knn_f1_macro], tick_label=['Stratified Dummy', 'k Nearest Neighbors'])
plt.show()

# ########### Step D ############
# 1) Oversample to fix imbalance
# 1.5) Try normalizing the data too, they mention it in a notbook
neighbors = list(range(1, 20, 2))
n_components = list(range(1,20, 1))

for n_neigh in neighbors:
    for n_comp in n_components:
        # 2) PCA with n components
        # 3) Create-fit-tranform knn with n_neigh neighbors
        # Find score, find best combo

        pass
x = 42

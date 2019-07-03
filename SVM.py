from sklearn.svm import SVR
from sklearn import svm
from Util import onehot, get_data, extract_features, split_by_ratio, feature_normalize, get_dataCSV, single_normalize, extract_moreSVM
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import csv
import pandas as pd

import sklearn.metrics as sm
import numpy as np

RATIO = 0.8

train, train_labels, test = get_dataCSV('train_labels.csv', 'train/', 'test/')
print('Forma datelor RAW:')
print(train.shape)
print(test.shape)


train = extract_moreSVM(train)
test = extract_moreSVM(test)
print('Dupa ce s-au extras feature-urile')
print(train.shape)
print(test.shape)

#train_labels, validation_labels = split_by_ratio(train_labels, RATIO)


svc = svm.SVC(C=100, kernel='poly', gamma='scale', degree=10)
svc.fit(train, train_labels)
scoruri_cv = cross_val_score(svc, train, train_labels, cv=3)
print('Scoruri de cross validare: ', scoruri_cv)
print('Scor cross validare mediu: {} +- {}'.format(scoruri_cv.mean(), scoruri_cv.std()*2))

predicted_test = np.array(svc.predict(test))
predicted_train = np.array(svc.predict(train))
print('Scor train: ')
print(sm.accuracy_score(predicted_train, train_labels))

print('Matrice de confuzie: ')
cm = np.array(confusion_matrix(train_labels, predicted_train))
print(cm)

sub = pd.DataFrame()
sample = pd.read_csv('./sample_submission.csv')
sub['id'] = sample['id']
sub['class'] = predicted_test.reshape((predicted_test.shape[0]))

sub.to_csv('Svm2Submission.csv', index=False)


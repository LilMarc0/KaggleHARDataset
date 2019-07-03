import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Util import *
from sklearn import metrics

RATIO = 0.8

train, train_labels, test = get_dataCSV('train_labels.csv', 'train/', 'test/')

train, validation = split_by_ratio(train, RATIO)
#train = single_normalize(train, 'l2')
#validation = single_normalize(validation, 'l2')
#test = single_normalize(test, 'l2')

train = extract_moreSVM(train)
test = extract_moreSVM(test)
validation = extract_moreSVM(validation)

train_labels, validation_labels = split_by_ratio(train_labels, RATIO)

train = feature_normalize(train, 'l2')
validation = feature_normalize(validation, 'l2')
test = feature_normalize(test, 'l2')

print(test.shape, train.shape, validation.shape)

rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, verbose=1, max_depth=150)
rf.fit(train, train_labels)
print(sorted(list(enumerate(rf.feature_importances_)), key=lambda x: x[1]))

validation_pred = rf.predict(validation)
print("Accuracy rf:", metrics.accuracy_score(validation_pred, validation_labels))

predicted_test = rf.predict(test)

sub = pd.DataFrame()
sample = pd.read_csv('./sample_submission.csv')
sub['id'] = sample['id']
# sub['class'] = predicted_test.reshape((predicted_test.shape[0])).astype(int)
#
# sub.to_csv('RandomForestSubmission.csv', index=False)
#np.savetxt('sub.csv', np.column_stack((sample['id'].tolist(), list(map(int, predicted_test)))), fmt='%s', delimiter=',', header='id,class', format='%d')
import numpy as np
from Util import *
from tensorflow import logging
logging.set_verbosity(logging.ERROR)
import pandas as pd

RATIO = 0.8
STEP = 15
SIZE = 30
N_EPOCHS = 1000
LR = 0.5

model_name = 'CNNFINAL'

train, train_labels, test = get_dataCSV('train_labels.csv', 'train/', 'test/')

# train = extract_moreSVM(train)
# test = extract_moreSVM(test)
# validation = extract_moreSVM(validation)
#
# train = feature_normalize(train)
# validation = feature_normalize(validation)
# test = feature_normalize(test)


# Cod folosit pentru extragerea feature-urilor din ferestre si serializarea lor
# ( dureaza foarte mult, le serializez pentru refolosire )
validation = extract_time_series(validation, STEP, SIZE, True)
pickle.dump(validation, open('./preprocessed/validation', 'wb'))
train = extract_time_series(train, STEP, SIZE, True)
pickle.dump(train, open('./preprocessed/train', 'wb'))
test = extract_time_series(test, STEP, SIZE, True)
pickle.dump(test, open('./preprocessed/test', 'wb'))

# Refolosire date
# train = np.array(pickle.load( open('./preprocessed/train', 'rb')))
# validation = np.array(pickle.load(open('./preprocessed/validation', 'rb')))
# test = np.array(pickle.load(open('./preprocessed/test', 'rb')))

print('---')
print(test.shape)
print(train.shape)
print(validation.shape)

# Reshape de tip (
train = np.reshape(train, (train.shape[0], train.shape[1],1, 8,8))
validation = np.reshape(validation, (validation.shape[0], validation.shape[1], 1, 8,8))
test = np.reshape(test, (test.shape[0], test.shape[1], 1, 8,8))

train_labels, validation_labels = split_by_ratio(train_labels, RATIO)

train_labels = onehot(train_labels)
validation_labels = onehot(validation_labels)

shape = np.array(train).shape
retea = cnn_lstm(1.0, LR, [None, shape[1], shape[2], shape[3], shape[4]])
#retea.load('./models/' + model_name + str(0.05))
print(train[0])
retea.fit(train, train_labels, batch_size=shape[0], n_epoch=N_EPOCHS, validation_set=(validation, validation_labels), show_metric=True, run_id='CNNLSTM2')
retea.save('./models/' + model_name + str(0.05))

predicted_test = np.array(list(map(np.argmax, retea.predict(test))))
#predicted_test = np.array(list(map(lambda x: x+1, predicted_test)))
print(predicted_test[:5], predicted_test.shape)

sub = pd.DataFrame()
sample = pd.read_csv('sample_submission.csv')
sub['id'] = sample['id']
sub['class'] = predicted_test.reshape((predicted_test.shape[0]))
print(sub.head(10))
sub.to_csv('./submissions/' + model_name + 'Submission' + '.csv', index=False)

#[(18, 0.01053329885558085), (16, 0.010536464716494288), (17, 0.010614179284121677), (19, 0.010674154522309676), (22, 0.010699926599937739), (26, 0.01070554537603189), (25, 0.010711010845037169), (23, 0.01071347508216208), (24, 0.010723899406035646), (27, 0.010729850198280513), (20, 0.010769839115785558), (21, 0.01081314575747545), (12, 0.011296220526623518), (3, 0.01227405753395234), (13, 0.012394775284873726), (6, 0.01575606925408744), (4, 0.01951172129040571), (8, 0.027880669446988693), (11, 0.03460808425064137), (5, 0.036881672270054296), (0, 0.048097087843452166), (7, 0.06653290381445734), (10, 0.07149250869858462), (9, 0.07740669253404213), (14, 0.08103096587315707), (15, 0.08852242488196328), (2, 0.1338457981606738), (1, 0.1342435585767899)]

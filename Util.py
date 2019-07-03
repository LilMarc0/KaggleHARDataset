import numpy as np
import os
from sklearn import preprocessing
import csv
import scipy.stats as st
import multiprocessing.dummy

def concatenate_csvs(path):
    concatenated = []
    for nume in os.listdir(path):
        data = np.genfromtxt(path + '/' + nume, delimiter=',')
        data = data[:134]

        if len(data) == 134:
            concatenated.append(data)
    return concatenated


def get_data(train_path, test_path):
    train = np.array(concatenate_csvs(train_path))
    test = np.array(concatenate_csvs(test_path))
    print('Train data shape: ' + str(train.shape))
    print('Test data shape' + str(test.shape))
    return np.concatenate(train).reshape((train.shape[0], 134, 3)),\
           np.concatenate(test).reshape((test.shape[0], 134, 3))

def get_dataCSV(label_path, train_path, test_path):
    train_data = []
    test_data = []
    train_labels = []
    train_csvs = os.listdir(train_path)
    test_csvs = os.listdir(test_path)

    # iterez in csv-uri si creez array-urile de train, test si labels
    for n in test_csvs:
        with open(test_path + n) as t:
            reader = csv.reader(t)
            datapoint = []
            for line in reader:
                datapoint.append(list(map(float, line)))
            for _ in range(160 - len(datapoint) % 160):
                datapoint.append([0.0, 0.0, 0.0])
        test_data.append(datapoint)

    with open(label_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            datapoint = []
            if row['id']+'.csv' in train_csvs:
                with open(train_path + row['id']+'.csv') as tr:
                    reader = csv.reader(tr)
                    for idx, line in enumerate(reader):
                        datapoint.append(list(map(float, line)))
                    #bordare
                    for _ in range(160-len(datapoint)%160):
                        datapoint.append([0.0,0.0,0.0])
                train_data.append(datapoint)
                train_labels.append(int(row['class']))

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    return train_data, train_labels, np.array(test_data)

# ratio: train/test
def split_by_ratio(data, ratio):
    # impart array-urile in train si validare
    return np.array(data[:int(len(data)*ratio)]), np.array(data[int(len(data)*ratio):])

# 1 - [1,0,0]; 2 - [0,1,0]; 3 - [0,0,1]
def onehot(data):

    one_hot_data = []
    for p in data:
        line = np.zeros((20), dtype='int')
        line[p-1] = 1
        one_hot_data.append(line)
    return np.array(one_hot_data)

# normalizeaza array-uri 3D
def single_normalize(x, type='l2'):

    sh = x.shape
    return preprocessing.scale(np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])), axis=0, with_std=True, with_mean=True).reshape(sh)

def cnn(eps, lr, shape):
    import tflearn
    inputs = tflearn.input_data(shape=shape)
    net = tflearn.conv_1d(inputs, nb_filter=64, filter_size=3, strides=1, padding='same', activation='tanh')
    net = tflearn.avg_pool_1d(net, kernel_size=3)
    net = tflearn.conv_1d(net, 64, 3, 1, padding='same', activation='tanh')
    shape = net.get_shape().as_list()
    print(shape)
    net = tflearn.reshape(net, [-1, shape[1] * shape[2]])

    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 512, activation='tanh', regularizer='L2')
    net = tflearn.dropout(net, 0.8)

    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 1024, activation='tanh', regularizer='L2')
    net = tflearn.dropout(net, 0.8)

    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 512, activation='tanh', regularizer='L2')

    softmax = tflearn.fully_connected(net, 20, activation='softmax')

    # sgd = tflearn.SGD(learning_rate=lr, lr_decay=0.96, decay_step=15)
    adam = tflearn.optimizers.adam(epsilon=eps, learning_rate=lr)
    regression = tflearn.regression(softmax, optimizer=adam, metric='accuracy', loss='categorical_crossentropy')

    model = tflearn.DNN(regression, tensorboard_verbose=3, tensorboard_dir='.', best_val_accuracy=0.7, best_checkpoint_path='./bestFinal/cnn', checkpoint_path='./checkpoints/cnn', max_checkpoints=10)
    print('Model created')
    return model

def cnn_lstm(eps, lr, shape):
    import tflearn
    inputs = tflearn.input_data(shape=shape)

    net = tflearn.time_distributed(inputs, tflearn.conv_2d, [32, 3, 1, 'same', 'tanh'])
    net = tflearn.time_distributed(net, tflearn.conv_2d, [32, 3, 1, 'same', 'tanh'])
    net = tflearn.time_distributed(net, tflearn.dropout, [0.8])

    net = tflearn.time_distributed(net, tflearn.avg_pool_2d, [2])

    print(net.get_shape().as_list())

    net = tflearn.reshape(net, [-1, net.get_shape().as_list()[1], net.get_shape().as_list()[2] * net.get_shape().as_list()[3] * net.get_shape().as_list()[4]])
    net = tflearn.lstm(net, 256, activation='tanh')
    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 256, activation='tanh', regularizer='L2')
    softmax = tflearn.fully_connected(net, 20, activation='softmax')

    # sgd = tflearn.SGD(learning_rate=lr, lr_decay=0.96, decay_step=15)
    adam = tflearn.optimizers.adam(epsilon=eps, learning_rate=lr)
    regression = tflearn.regression(softmax, optimizer=adam, metric='accuracy')

    model = tflearn.DNN(regression, tensorboard_verbose=3, tensorboard_dir='.',  best_val_accuracy=0.6,checkpoint_path='./checkpoints2/cnn2', max_checkpoints=1)
    return model

def lstm(eps, lr, shape):
    import tflearn
    inputs = tflearn.input_data(shape=shape)

    net = tflearn.lstm(inputs, 128, return_seq=True, activation='tanh')
    net = tflearn.dropout(net, 0.9)

    net = tflearn.lstm(net, 256, return_seq=True, activation='tanh')
    net = tflearn.dropout(net, 0.9)

    net = tflearn.lstm(net, 128, activation='tanh')

    softmax = tflearn.fully_connected(net, 20, activation='softmax')

    # sgd = tflearn.SGD(learning_rate=lr, lr_decay=0.96, decay_step=15)
    adam = tflearn.optimizers.adam(epsilon=eps, learning_rate=lr)
    regression = tflearn.regression(softmax, optimizer=adam, metric='accuracy')

    model = tflearn.DNN(regression, tensorboard_verbose=3, tensorboard_dir='.')
    print('Model created')
    return model

def nn(eps, lr, shape):
    import tflearn
    inputs = tflearn.input_data(shape=shape)

    net = tflearn.fully_connected(inputs, 512, activation='tanh', regularizer='L2')
    net = tflearn.dropout(net, 0.9)

    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 1024, activation='tanh', regularizer='L2')
    net = tflearn.dropout(net, 0.9)

    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 512, activation='tanh', regularizer='L2')

    softmax = tflearn.fully_connected(net, 20, activation='softmax')

    #sgd = tflearn.SGD(learning_rate=lr, lr_decay=0.96, decay_step=15)
    adam = tflearn.optimizers.adam(epsilon=eps, learning_rate=lr)
    regression = tflearn.regression(softmax, optimizer=adam, metric='accuracy')

    model = tflearn.DNN(regression, tensorboard_verbose=3, tensorboard_dir='.')
    print('Model created')
    return model

def feature_normalize(train_data, type='l2'):
    # normalizeaza array-uri 2D
    train_data = preprocessing.scale(train_data, axis = 0, with_std=True, with_mean=True)
    return train_data

# X - array-ul pe care il despart in ferestre
# step - cate pozitii se sar
# size - marimea ferestrei
# features - daca este True, se extrag features din ferestre in timp ce se formeaza
def extract_time_series(X, step, size,features=False):
    tdataset = []
    p = multiprocessing.dummy.Pool(8)
    for idx, d in enumerate(X):
        # if idx % 100 == 0:
        #     print(idx)
        windows = get_windows(d, step, size)
        if features:
            if idx % 100 == 0:
                print(idx)
            tdataset.append(list(map(extract_more, windows)))
        else:
            tdataset.append(windows)
    return np.array(tdataset)

# impartirea efectiva in ferestre
def get_windows(seq, step, size):
    windows = []
    iter = 0
    lines = 0
    while iter + size < len(seq):
        windows.append(seq[iter : iter + size])
        iter += step
        lines += 1
    windows.append(seq[iter : len(seq)])
    # print(windows[-1])
    for _ in range(size - len(windows[-1])%size):
        # print('-------')
        # print(len(windows[-1]))
        # print(windows[-1])
        windows[-1] = np.append(windows[-1], [[0.0, 0.0, 0.0]], axis=0)
    return windows

# NU SE MAI FOLOSESTE
# Extragere de features din array 2D
def extract_features_single(dataset):
    fdataset = np.zeros((16))

    from scipy.fftpack import diff, fft

    for i in range(len(dataset)):
        # x             y               z
        fdataset[0], fdataset[1], fdataset[2] = np.mean(dataset, axis=0)  # mean
        fdataset[3], fdataset[4], fdataset[5] = np.std(dataset, axis=0)  # stddev
        fdataset[6], fdataset[7], fdataset[8] = np.min(dataset, axis=0)  # min
        fdataset[9], fdataset[10], fdataset[11] = np.max(dataset, axis=0)  # max

        # corr
        fdataset[12] = sum((dataset[:, 0] - fdataset[0]) * (dataset[:, 1] - fdataset[1])) / sum(
            (dataset[:, 0] - fdataset[0]) ** 2 * (dataset[:, 1] - fdataset[1]) ** 2)
        fdataset[13] = sum((dataset[:, 0] - fdataset[0]) * (dataset[:, 2] - fdataset[2])) / sum(
            (dataset[:, 0] - fdataset[0]) ** 2 * (dataset[:, 2] - fdataset[2]) ** 2)
        fdataset[14] = sum((dataset[:, 2] - fdataset[2]) * (dataset[:, 1] - fdataset[1])) / sum(
            (dataset[:, 2] - fdataset[2]) ** 2 * (dataset[:, 1] - fdataset[1]) ** 2)

        fftdata = np.fft.fft(dataset[:])
        fdataset[15] = np.sum(np.sqrt(fftdata.real ** 2 + fftdata.imag ** 2))

        # mag
        fdataset[16] = np.sqrt(np.sum(fdataset[0], fdataset[1], fdataset[2]))

        # jerk
        fdataset[17] = diff(dataset[:, 0])

    return np.array(fdataset)

# NU SE MAI FOLOSESTE
# Extragere features din array 3D
def extract_features(dataset):
    fdataset = np.zeros((len(dataset), 16))

    for i in range(len(dataset)):
        # x             y               z
        fdataset[i][0], fdataset[i][1], fdataset[i][2] = np.mean(dataset[i], axis=0) # mean
        fdataset[i][3], fdataset[i][4], fdataset[i][5] = np.std(dataset[i], axis=0)  # stddev
        fdataset[i][6], fdataset[i][7], fdataset[i][8] = np.min(dataset[i], axis =0) # min
        fdataset[i][9], fdataset[i][10], fdataset[i][11] = np.max(dataset[i], axis=0)# max

        # corr
        fdataset[i][12] = sum( (dataset[i, :, 0] - fdataset[i][0]) * (dataset[i, :, 1] - fdataset[i,1]) ) / sum(
            (dataset[i, :, 0] - fdataset[i][0])**2 * (dataset[i, :, 1] - fdataset[i,1])**2 )
        fdataset[i][13] = sum((dataset[i, :, 0] - fdataset[i][0]) * (dataset[i, :, 2] - fdataset[i, 2])) / sum(
            (dataset[i, :, 0] - fdataset[i][0]) ** 2 * (dataset[i, :, 2] - fdataset[i, 2]) ** 2)
        fdataset[i][14] = sum((dataset[i, :, 2] - fdataset[i][2]) * (dataset[i, :, 1] - fdataset[i, 1])) / sum(
            (dataset[i, :, 2] - fdataset[i][2]) ** 2 * (dataset[i, :, 1] - fdataset[i, 1]) ** 2)

        fftdata = np.fft.fft(dataset[i, :])
        fdataset[i][15] = np.sum(np.sqrt(fftdata.real**2 + fftdata.imag**2))


    return np.array(fdataset)

# Extrage 64 de features din array 2D
def extract_more(p):

    data = []
    data.append(np.mean(p[0]))
    data.append(np.mean(p[1]))
    data.append(np.mean(p[2]))
    data.append(np.std(p[0]))
    data.append(np.std(p[1]))
    data.append(np.std(p[2]))
    data.append(np.min(p[0]))
    data.append(np.min(p[1]))
    data.append(np.min(p[2]))
    data.append(np.max(p[0]))
    data.append(np.max(p[1]))
    data.append(np.max(p[2]))
    data.append(len(np.where(np.diff(np.sign(p[  0])))[0])) # 0 cross
    data.append(len(np.where(np.diff(np.sign(p[1])))[0]))
    data.append(len(np.where(np.diff(np.sign(p[2])))[0]))
    data.append(st.pearsonr(p[  0], p[1])[0])
    data.append(st.pearsonr(p[  0], p[1])[1])
    data.append(st.pearsonr(p[0], p[2])[0])
    data.append(st.pearsonr(p[0], p[2])[1])
    data.append(st.pearsonr(p[2], p[1])[0])
    data.append(st.pearsonr(p[2], p[1])[1])
    data.append(st.variation(p[0]))
    data.append(st.variation(p[1]))
    data.append(st.variation(p[2]))
    data.append(np.mean(np.gradient(p[0])))
    data.append(np.mean(np.gradient(p[1])))
    data.append(np.mean(np.gradient(p[2])))
    data.append(np.min(np.gradient(p[0])))
    data.append(np.min(np.gradient(p[1])))
    data.append(np.min(np.gradient(p[2])))
    data.append(np.max(np.gradient(p[0])))
    data.append(np.max(np.gradient(p[1])))
    data.append(np.max(np.gradient(p[2])))
    data.append(np.std(np.gradient(p[0])))
    data.append(np.std(np.gradient(p[1])))
    data.append(np.std(np.gradient(p[2])))
    data.append(np.mean(np.sin(p[0])))
    data.append(np.mean(np.sin(p[1])))
    data.append(np.mean(np.sin(p[2])))
    data.append(np.mean(np.cos(p[0])))
    data.append(np.mean(np.cos(p[1])))
    data.append(np.mean(np.cos(p[2])))
    data.append(np.min(np.cos(p[0])))
    data.append(np.min(np.cos(p[1])))
    data.append(np.min(np.cos(p[2])))
    data.append(np.min(np.sin(p[0])))
    data.append(np.min(np.sin(p[1])))
    data.append(np.min(np.sin(p[2])))
    data.append(np.max(np.sin(p[0])))
    data.append(np.max(np.sin(p[1])))
    data.append(np.max(np.sin(p[2])))
    data.append(np.max(np.cos(p[0])))
    data.append(np.max(np.cos(p[1])))
    data.append(np.max(np.cos(p[2])))
    data.append(np.std(np.cos(p[0])))
    data.append(np.std(np.cos(p[1])))
    data.append(np.std(np.cos(p[2])))
    data.append(np.std(np.sin(p[0])))
    data.append(np.std(np.sin(p[1])))
    data.append(np.std(np.sin(p[2])))
    fftdata = np.fft.fft(p)
    fftx = np.fft.fft(p[0])
    ffty = np.fft.fft(p[1])
    fftz = np.fft.fft(p[2])
    data.append(np.sum(np.sqrt(fftdata.real ** 2 + fftdata.imag ** 2)))
    data.append(np.sum(np.sqrt(fftx.real ** 2 + fftx.imag ** 2)))
    data.append(np.sum(np.sqrt(ffty.real ** 2 + ffty.imag ** 2)))
    data.append(np.sum(np.sqrt(fftz.real ** 2 + fftz.imag ** 2)))
    return np.array(data)

# Extrage 64 de features din array 3D
def extract_moreSVM(dataset):
    newData = []
    for p in dataset:
        data = []
        data.append(np.mean(p[:,0]))
        data.append(np.mean(p[:,1]))
        data.append(np.mean(p[:,2]))
        data.append(np.std(p[:,0]))
        data.append(np.std(p[:,1]))
        data.append(np.std(p[:,2]))
        data.append(np.min(p[:,0]))
        data.append(np.min(p[:,1]))
        data.append(np.min(p[:,2]))
        data.append(np.max(p[:,0]))
        data.append(np.max(p[:,1]))
        data.append(np.max(p[:,2]))
        data.append(len(np.where(np.diff(np.sign(p[:,0])))[0])) # 0 cross
        data.append(len(np.where(np.diff(np.sign(p[:,1])))[0]))
        data.append(len(np.where(np.diff(np.sign(p[:,2])))[0]))
        data.append(st.pearsonr(p[  :,0], p[:,1])[0])
        data.append(st.pearsonr(p[  :,0], p[:,1])[1])
        data.append(st.pearsonr(p[:,0], p[:,2])[0])
        data.append(st.pearsonr(p[:,0], p[:,2])[1])
        data.append(st.pearsonr(p[:,2], p[:,1])[0])
        data.append(st.pearsonr(p[:,2], p[:,1])[1])
        data.append(st.variation(p[:,0]))
        data.append(st.variation(p[:,1]))
        data.append(st.variation(p[:,2]))
        data.append(np.mean(np.gradient(p[:,0])))
        data.append(np.mean(np.gradient(p[:,1])))
        data.append(np.mean(np.gradient(p[:,2])))
        data.append(np.min(np.gradient(p[:,0])))
        data.append(np.min(np.gradient(p[:,1])))
        data.append(np.min(np.gradient(p[:,2])))
        data.append(np.max(np.gradient(p[:,0])))
        data.append(np.max(np.gradient(p[:,1])))
        data.append(np.max(np.gradient(p[:,2])))
        data.append(np.std(np.gradient(p[:,0])))
        data.append(np.std(np.gradient(p[:,1])))
        data.append(np.std(np.gradient(p[:,2])))
        data.append(np.mean(np.sin(p[:,0])))
        data.append(np.mean(np.sin(p[:,1])))
        data.append(np.mean(np.sin(p[:,2])))
        data.append(np.mean(np.cos(p[:,0])))
        data.append(np.mean(np.cos(p[:,1])))
        data.append(np.mean(np.cos(p[:,2])))
        data.append(np.min(np.cos(p[:,0])))
        data.append(np.min(np.cos(p[:,1])))
        data.append(np.min(np.cos(p[:,2])))
        data.append(np.min(np.sin(p[:,0])))
        data.append(np.min(np.sin(p[:,1])))
        data.append(np.min(np.sin(p[:,2])))
        data.append(np.max(np.sin(p[:,0])))
        data.append(np.max(np.sin(p[:,1])))
        data.append(np.max(np.sin(p[:,2])))
        data.append(np.max(np.cos(p[:,0])))
        data.append(np.max(np.cos(p[:,1])))
        data.append(np.max(np.cos(p[:,2])))
        data.append(np.std(np.cos(p[:,0])))
        data.append(np.std(np.cos(p[:,1])))
        data.append(np.std(np.cos(p[:,2])))
        data.append(np.std(np.sin(p[:,0])))
        data.append(np.std(np.sin(p[:,1])))
        data.append(np.std(np.sin(p[:,2])))
        fftdata = np.fft.fft(p)
        fftx = np.fft.fft(p[:,0])
        ffty = np.fft.fft(p[:,1])
        fftz = np.fft.fft(p[:,2])
        data.append(np.sum(np.sqrt(fftdata.real ** 2 + fftdata.imag ** 2)))
        data.append(np.sum(np.sqrt(fftx.real ** 2 + fftx.imag ** 2)))
        data.append(np.sum(np.sqrt(ffty.real ** 2 + ffty.imag ** 2)))
        data.append(np.sum(np.sqrt(fftz.real ** 2 + fftz.imag ** 2)))
        newData.append(data)
    return np.array(newData)
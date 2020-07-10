import warnings
warnings.filterwarnings("ignore")
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, MaxPooling2D, Conv2D, Concatenate, Flatten
from librosa import load
import os
from pydub import AudioSegment
from pydub.silence import detect_silence
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from python_speech_features import mfcc
from librosa.feature import delta
from sys import argv
import time
from scipy.stats import mode

# Data setup
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

train_dir = 'emotion-dataset/'
user_test_dir = 'user-test/'
data_from_previous = int(argv[1])
use_saved_model = int(argv[2])
user_test = int(argv[3])

x_train_filename = 'x_train_data.npy'
y_train_filename = 'y_train_data.npy'
model_path = 'nn.model'

classes = os.listdir(train_dir)
n_classes = len(classes)
corpus_breakdown = [0 for i in range(n_classes)]

# For truncating silence (ms)
min_silence_len = 500
silence_thresh= -10

# MFCC/dataset params
num_cep = 20
num_feats = num_cep*3
win_step = 0.025
win_len = 0.05
nfft = 2048

# For formatting dataset
stack_length = 64
n_data = 0
size_list = 0
x_train_list = []
y_label_list = []

if data_from_previous:
    print("\n==========\nLoading previous dataset from {0} and {1}...\n==========".format(x_train_filename, y_train_filename))
    x_train = np.load(x_train_filename)
    y_train = np.load(y_train_filename)

else:
    print("\n==========\nProcessing dataset from ({0}) directory...\n==========".format(train_dir))
    for i in range(n_classes):
        print("\n==========\nProcessing files for class: ({0})\n==========".format(classes[i]))
        filepath = train_dir+classes[i]+'/'
        train_files = os.listdir(filepath)
        for fname in train_files:
            train_path = filepath+fname
            signal, sample_rate = load(train_path,sr=None)

            signal_for_silence = AudioSegment.from_file(train_path,format='wav')
            silence_indices = detect_silence(signal_for_silence,min_silence_len=min_silence_len,silence_thresh=silence_thresh)
            signal = np.delete(signal, silence_indices)

            mfcc_feats = mfcc(signal=signal,numcep=num_cep,samplerate=sample_rate,winstep=win_step,winfunc=np.hamming,nfft=nfft)
            delta_feats = delta(data=mfcc_feats,order=1)
            delta2_feats = delta(data=mfcc_feats,order=2)

            if mfcc_feats.shape[0] < stack_length:
                print("\n==========\nDEBUG: Excluded file {0} because feature length is too short after silence truncation (length was {1}).\n==========".format(train_path,mfcc_feats.shape[0]))

            else:
                corpus_breakdown[i] += 1
                features = np.zeros((mfcc_feats.shape[0],num_feats,1))
                features[:,0:num_cep,0] = mfcc_feats
                features[:,num_cep:2*num_cep,0] = delta_feats
                features[:,2*num_cep:3*num_cep,0] = delta2_feats

                labels = np.zeros((mfcc_feats.shape[0],n_classes))
                labels[:,i] = 1

                x_train_list.append(features)
                y_label_list.append(labels)
                n_data += mfcc_feats.shape[0]
                size_list += 1

    n_train_data = n_data // stack_length
    x_train = np.zeros((n_train_data, stack_length, num_feats, 1))
    y_train = np.zeros((n_train_data,n_classes))

    train_index = 0
    data_index = 0
    for i in range(size_list):
        len_data = x_train_list[i].shape[0]
        len_data //= stack_length
        data_index = 0
        y_train[train_index:train_index+len_data] = (y_label_list[i])[0:len_data]
        for datum in range(len_data):
            x_train[train_index,:,:] = (x_train_list[i])[data_index:data_index+stack_length]
            train_index += 1
            data_index += stack_length

    x_train_trunc = x_train[0:train_index]
    y_train_trunc = y_train[0:train_index]

    x_train = x_train_trunc
    y_train = y_train_trunc

    print("\n==========\nDataset corpus breakdown per class:")
    for i in range(n_classes):
        print("Class {0} has {1} files used in dataset".format(classes[i],corpus_breakdown[i]))
    print("==========")

    np.save(x_train_filename,x_train)
    np.save(y_train_filename,y_train)


# Generating batch sizes and validation set
batch_size = 32
validation_split = 0.2

size_validation = int(x_train.shape[0] * validation_split)
size_train = x_train.shape[0] - size_validation

x_train, y_train = shuffle(x_train, y_train)

x_test = x_train[-size_validation:]
y_test = y_train[-size_validation:]

x_train = x_train[:-size_validation]
y_train = y_train[:-size_validation]

num_batches = size_train // batch_size
num_val_batches = size_validation // batch_size

input_shape = (stack_length, num_feats, 1) # (num_data, num_stacked, mfcc_feats)

# NN
if use_saved_model:
    print("\n==========\nUsing previous model from {0}...\n==========".format(model_path))
    model = tf.keras.models.load_model(model_path)

else:
    print("\n==========\nTraining new model...\n==========")

    # Training/arch parameters
    epochs = 200

    filter_size1 = 50
    kernel_size1 = (4,6)

    filter_size2 = 50
    kernel_size2 = (6,8)

    filter_size3 = 50
    kernel_size3 = (8,10)

    filter_size4 = 50
    kernel_size4 = (10,12)

    fc1_size = 400
    fc2_size = 200

    dropout_rate = 0.5

    input = Input(shape=input_shape)

    conv1 = Conv2D(filter_size1, kernel_size1, activation='relu')(input)
    conv2 = Conv2D(filter_size2, kernel_size2, activation='relu')(input)
    conv3 = Conv2D(filter_size3, kernel_size3, activation='relu')(input)
    conv4 = Conv2D(filter_size4, kernel_size4, activation='relu')(input)

    pool_size1 = (int(conv1.shape[1]//2), int(conv1.shape[2]//2))
    pool1 = MaxPooling2D(pool_size1)(conv1)
    pool_size2 = (int(conv2.shape[1]//2), int(conv2.shape[2]//2))
    pool2 = MaxPooling2D(pool_size2)(conv2)
    pool_size3 = (int(conv3.shape[1]//2), int(conv3.shape[2]//2))
    pool3 = MaxPooling2D(pool_size3)(conv3)
    pool_size4 = (int(conv4.shape[1]//2), int(conv4.shape[2]//2))
    pool4 = MaxPooling2D(pool_size4)(conv4)

    pool_merge = Concatenate()([pool1,pool2,pool3,pool4])
    flat_pooling = Flatten()(pool_merge)

    fc1 = Dense(fc1_size,activation='relu')(flat_pooling)
    drop = Dropout(rate=dropout_rate)(fc1)

    fc2 = Dense(fc2_size)(drop)

    output = Dense(n_classes,activation='softmax')(fc2)

    model = Model(inputs=input,outputs=output)

    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()

    print("\n==========\nTrain on {0} samples, validate on {1} samples, over {2} epochs, with batch size {3}.\n==========".format(size_train, size_validation, epochs, batch_size))

    for epoch in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)
        loss, acc = 0.0, 0.0
        for batch_index in range(num_batches):
            x_batch, y_batch = x_train[batch_index*batch_size:(batch_index+1)*batch_size], y_train[batch_index*batch_size:(batch_index+1)*batch_size]
            metrics = model.train_on_batch(x_batch, y_batch)
            loss += metrics[0]
            acc += metrics[1]

        model.save(model_path)

        x_test, y_test = shuffle(x_test, y_test)
        val_loss, val_acc = 0.0, 0.0
        for batch_index in range(num_val_batches):
            x_test_batch, y_test_batch = x_test[batch_index*batch_size:(batch_index+1)*batch_size], y_test[batch_index*batch_size:(batch_index+1)*batch_size]
            metrics = model.test_on_batch(x_batch, y_batch)
            val_loss += metrics[0]
            val_acc += metrics[1]

        print("\n==========\nEpoch: {0} -- Avg Loss: {1:.4f} -- Avg Acc: {2:.4f} -- Val Loss: {3:.4f} -- Val Acc: {4:.4f}\n==========".format(epoch, loss/num_batches, acc/num_batches, val_loss/num_val_batches, val_acc/num_val_batches))

# Specific user file test
if user_test:
    test_files = os.listdir(user_test_dir)
    # Test each file individually
    for file in test_files:
        test_path = user_test_dir+file
        signal, sample_rate = load(test_path, sr=None)

        signal_for_silence = AudioSegment.from_file(test_path,format='wav')
        silence_indices = detect_silence(signal_for_silence,min_silence_len=min_silence_len,silence_thresh=silence_thresh)
        signal = np.delete(signal, silence_indices)

        mfcc_feats = mfcc(signal=signal,numcep=num_cep,samplerate=sample_rate,winstep=win_step,winfunc=np.hamming,nfft=nfft)
        delta_feats = delta(data=mfcc_feats,order=1)
        delta2_feats = delta(data=mfcc_feats,order=2)

        if mfcc_feats.shape[0] < stack_length:
            print("\n==========\nDEBUG: Excluded test file {0} because feature length is too short after silence truncation (length was {1}).\n==========".format(test_path,mfcc_feats.shape[0]))

        else:
            features = np.zeros((mfcc_feats.shape[0],num_feats,1))
            features[:,0:num_cep,0] = mfcc_feats
            features[:,num_cep:2*num_cep,0] = delta_feats
            features[:,2*num_cep:3*num_cep,0] = delta2_feats

            len_data = mfcc_feats.shape[0]
            len_data //= stack_length
            x_test = np.zeros((len_data,stack_length,num_feats,1))

            for datum in range(len_data):
                x_test[datum,:,:] = features[datum*stack_length:(datum+1)*stack_length]

            prediction = np.argmax(model.predict_on_batch(x_test))

            print("\n==========\nPredicted class {0} for file {1}\n==========".format(classes[prediction],test_path))


else:
    confusion_matrix = np.zeros((n_classes,n_classes))
    label_frequency = np.zeros(n_classes)
    # Test each datum individually
    x_test, y_test = shuffle(x_test, y_test)
    num_correct = 0.0
    num_total = float(size_validation)
    for i in range(size_validation):
        datum = x_test[i].reshape(1,input_shape[0],input_shape[1],input_shape[2])
        prediction = model.predict_on_batch(datum)
        y_hat = np.argmax(prediction)
        y = np.argmax(y_test[i])

        if y == y_hat:
            num_correct += 1.0

        label_frequency[y] += 1
        confusion_matrix[y_hat,y] += 1

    for i in range(n_classes):
        confusion_matrix[i,:] /= label_frequency[i]

    accuracy = num_correct/num_total
    confusion_matrix = confusion_matrix.round(decimals=2)
    print("\n==========\nFinal Accuracy: {0:.4f}\n==========".format(accuracy))
    print("\n==========\nConfusion matrix:\n{0}\n==========".format(np.matrix(confusion_matrix)))
    print("\n==========\nClass map for reference:")
    for i in range(n_classes):
        print(i,classes[i])
    print("==========")

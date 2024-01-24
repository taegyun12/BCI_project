import numpy as np
import scipy
from scipy.signal import filtfilt
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import regularizers
from preprocssing import pre_process
from itr import itr

#0번 gpu 사용
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU')

# Preliminaries
subban_no = 3
dataset = 'Bench'
signal_length = 0.4

if dataset == 'Bench':
    totalsubject = 35
    totalblock = 6
    totalcharacter = 4
    sampling_rate = 250
    visual_latency = 0.14
    visual_cue = 0.5
    sample_length = int(sampling_rate * signal_length)
    total_ch = 64
    max_epochs = 1000
    dropout_second_stage = 0.6

total_delay = visual_latency + visual_cue
delay_sample_point = round(total_delay * sampling_rate)
sample_interval = list(range(delay_sample_point, delay_sample_point + sample_length))#(python방식 인덱싱)
channels = np.array([54, 55, 56, 57, 58, 61, 62, 63])  # Indexes of 8 channels(python방식 인덱싱)



#PreProcessing
AllData, y_AllData = pre_process(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset)
acc_matrix = np.zeros((totalsubject, totalblock))
sizes = AllData.shape
#    AllData = np.zeros((total_channels, sample_length, subban_no, totalcharacter, totalblock, totalsubject))
#   y_AllData = np.zeros((1, totalcharacter, totalblock, totalsubject))

# Training and Testing
for block in range(totalblock):
   #한 블럭 씩 제외
    allblock = list(range(totalblock))
    allblock.remove(block)

    # Create and compile model
    model1 = Sequential()
    model1.add(InputLayer(input_shape=(sizes[0], sizes[1], sizes[2])))
    model1.add(Conv2D(1, (1,1), kernel_initializer='ones', use_bias=False))
    model1.add(Conv2D(120, (sizes[0], 1), kernel_initializer='random_normal'))
    model1.add(Dropout(0.1))
    model1.add(Conv2D(120, (1,2), strides=(1,2), kernel_initializer='random_normal'))
    model1.add(Dropout(0.1))
    model1.add(ReLU())
    model1.add(Conv2D(120, (1,10), padding='same', kernel_initializer='random_normal'))
    model1.add(Dropout(0.95))
    model1.add(Flatten())
    model1.add(Dense(totalcharacter, kernel_initializer='random_normal',
             kernel_regularizer=keras.regularizers.l2(0.001)))
    model1.add(Softmax())


    train = AllData[:, :, :, :, allblock, :]
    train = np.reshape(train, (sizes[0], sizes[1], sizes[2], totalcharacter * len(allblock) * totalsubject))

    train_y = y_AllData[:, :, allblock, :]
    train_y = np.reshape(train_y, (totalcharacter * len(allblock) * totalsubject))

    # First stage training
    optimizer = Adam(learning_rate=0.0001)
    callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

    model1.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Assuming it's a classification task
                  metrics=['accuracy'])

    train = np.transpose(train, (3,0,1,2))
    history = model1.fit(train, train_y,
                        epochs=max_epochs,
                        batch_size=100,
                        shuffle=True,
                        verbose=1,
                        callbacks=[callback])

    # Save the trained model1
    sv_name = f'main_net_{block}.h5'
    model1.save(sv_name)

    # Initialization of confusion matrixs
    all_conf_matrix = tf.cast(tf.zeros((4, 4)),tf.int32)
    # Second stage training
    for s in range(1, totalsubject + 1):
        model = Sequential([
            InputLayer(input_shape=(sizes[0], sizes[1], sizes[2])),
            Conv2D(1, (1,1), use_bias=False),
            Conv2D(120, (sizes[0], 1)),
            Dropout(dropout_second_stage),
            Conv2D(120, (1,2), strides=(1,2)),
            Dropout(dropout_second_stage),
            ReLU(),
            Conv2D(120, (1,10), padding='same'),
            Dropout(0.95),
            Flatten(),
            Dense(totalcharacter,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
            Softmax(),
])


        model.layers[0].set_weights(model1.layers[0].get_weights())
        model.layers[1].set_weights(model1.layers[1].get_weights())
        model.layers[3].set_weights(model1.layers[3].get_weights())
        model.layers[6].set_weights(model1.layers[6].get_weights())
        model.layers[9].set_weights(model1.layers[9].get_weights())


        model.layers[1].bias = model1.layers[1].bias
        model.layers[3].bias = model1.layers[3].bias
        model.layers[6].bias = model1.layers[6].bias
        model.layers[9].bias = model1.layers[9].bias

        #subject-specific data
        train = AllData[:, :, :, :, allblock, s - 1]
        train = np.reshape(train, (sizes[0], sizes[1], sizes[2], totalcharacter * len(allblock)))

        train_y = y_AllData[:, :, allblock, s - 1]
        train_y = np.reshape(train_y, (totalcharacter * len(allblock)))

        testdata = AllData[:, :, :, :, block, s - 1]
        testdata = np.reshape(testdata, (sizes[0], sizes[1], sizes[2], totalcharacter))

        test_y = y_AllData[:, :, block, s - 1]
        test_y = np.reshape(test_y, (totalcharacter))


        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',  # Assuming it's a classification task
                      metrics=['accuracy'],
                      )

        train = np.transpose(train, (3,0,1,2))

        history = model.fit(train, train_y,
                            epochs=1000,
                            batch_size=totalcharacter*(totalblock-1),
                            shuffle=True,
                            verbose=1)
        testdata = np.transpose(testdata, (3,0,1,2))
        YPred = model.predict(testdata)

        acc = np.mean(np.argmax(YPred, axis=1) == test_y)
        acc_matrix[s - 1, block] = acc

        print(type(all_conf_matrix))
        all_conf_matrix += tf.math.confusion_matrix(tf.cast(test_y, tf.int32), tf.cast(np.argmax(YPred, axis=1), tf.int32))


    sv_name = f'confusion_mat_{block+1}.npy'
    np.save(sv_name, all_conf_matrix.numpy())  # TensorFlow tensor를 NumPy 배열로 변환하여 저장

sv_name = 'acc_matrix.npy'
np.save(sv_name, acc_matrix)

itr_matrix = itr(acc_matrix, totalcharacter, visual_cue + signal_length)

print(itr_matrix)

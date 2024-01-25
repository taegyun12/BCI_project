import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from wearable_PreProcessing import pre_process
import tensorflow as tf
from itr import itr
#Specifications
subban_no = 3 # the number of subbands == bandpass filters
dataset = 'Bench' # Bench or BETA dataset
signal_length = 0.4 #Signal length in second

if dataset == "Bench":
    totalsubject = 50 # the number of subjects
    totalblock = 6 # the number of blocks
    totalcharacter = 12 # the number of characters
    sampling_rate = 250 # Sampling Rate
    visual_latency = 0.14 # Average visual latency of subjects
    visual_cue = 0.5 # Length of visual cue used at collection of the dataset
    sample_length = int(sampling_rate * signal_length) # Sample length
    total_ch = 64 # the number of channels used at collection of the dataset
    max_epochs = 800 # the number of epochs for first stage  ORIGINAL VALUE = 1000 !!!
    dropout_second_stage = 0.6 #Dropout probabilities of first two dropout layers at second stage
elif dataset == "BETA":
    total_subject = 70  # the number of subjects
    total_block = 4  # the number of blocks
    total_character = 40  # the number of characters
    sampling_rate = 250  # Sampling Rate
    visual_latency = 0.13  # Average visual latency of subjects
    visual_cue = 0.5  # Length of visual cue used at collection of the dataset
    sample_length = int(sampling_rate * signal_length)  # Sample length
    total_ch = 64  # the number of channels used at collection of the dataset
    max_epochs = 800  # the number of epochs for first stage
    dropout_second_stage = 0.7  # Dropout probabilities of first two dropout layers at second stage

# Preprocessing
total_delay = visual_latency + visual_cue # Total undesired signal length in seconds
delay_sample_point = round(total_delay * sampling_rate) # the number of data points correspond for undesired signal length
sample_interval = list(range(delay_sample_point, delay_sample_point + sample_length)) # Extract desired signal
# channels = [48, 54, 55, 56, 57, 58, 61, 62, 63] # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
channels =  np.array([1, 2, 3, 4, 5, 6,  7, 8]) # Indexes of 8 channels: (PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)


# To use all the channels set channels to 1:total_ch=64
AllData, y_AllData = pre_process(channels, sample_length, sample_interval, subban_no, totalsubject, totalblock, totalcharacter, sampling_rate, dataset)

# Dimension of AllData: (# of channels, # sample length, #subbands, # of characters, # of blocks, # of subjects)

#Evaluation
acc_matrix = np.zeros((totalsubject, totalblock)) #Initialization of accuracy matrix
sizes = AllData.shape ## 일단 ALLData가 numpy 배열이라 가정

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

    #model1.layers[1].bias.trainable = False

    train = AllData[:, :, :, :, allblock, :]
    train = np.reshape(train, (sizes[0], sizes[1], sizes[2], totalcharacter * len(allblock) * totalsubject))

    train_y = y_AllData[:, :, allblock, :]
    train_y = np.reshape(train_y, (totalcharacter * len(allblock) * totalsubject))
    #train_y = np.asarray(train_y, dtype=np.int32)
    #train_y = tf.keras.utils.to_categorical(train_y)  # Convert labels to one-hot encoding

    # print(train.shape)
    # print(train_y.shape)

    # First stage training
    optimizer = Adam(learning_rate=0.0001)
    callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

    model1.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Assuming it's a classification task
                  metrics=['accuracy'])

    train = np.transpose(train, (3,0,1,2))
    # print(train)
    history = model1.fit(train, train_y,
                        epochs=max_epochs,
                        batch_size=100,
                        shuffle=True,
                        verbose=1,
                        callbacks=[callback])

    # Save the trained model1
    sv_name = f'main_net_{block}.h5'
    model1.save(sv_name)

    # Initialization of confusion matrix
    # all_conf_matrix = tf.cast(tf.zeros((40, 40)),tf.int32)
    all_conf_matrix = tf.cast(tf.zeros((12, 12)),tf.int32)
    # Second stage training
    for s in range(1, totalsubject + 1):
        print('block: ', block)
        print('subject num: ', s)
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
        # print(model1.layers)

        model.layers[0].set_weights(model1.layers[0].get_weights())
        model.layers[1].set_weights(model1.layers[1].get_weights())
        model.layers[3].set_weights(model1.layers[3].get_weights())
        model.layers[6].set_weights(model1.layers[6].get_weights())
        model.layers[9].set_weights(model1.layers[9].get_weights())

        #model[1].bias.trainable = False
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

        #test_y = np.asarray(test_y, dtype=np.int32)
        #test_y = tf.keras.utils.to_categorical(test_y)
        #train_y = np.asarray(train_y, dtype=np.int32)
        #train_y = tf.keras.utils.to_categorical(train_y)

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',  # Assuming it's a classification task
                      metrics=['accuracy'],
                      )

        train = np.transpose(train, (3,0,1,2))
        # print(train.shape)
        # print(train_y.shape)
        history = model.fit(train, train_y,
                            epochs=400, # original epochs = 1000
                            batch_size=totalcharacter*(totalblock-1),
                            shuffle=True,
                            verbose=1)
        testdata = np.transpose(testdata, (3,0,1,2))
        YPred = model.predict(testdata)

        #print(np.argmax(YPred, axis=1))
        acc = np.mean(np.argmax(YPred, axis=1) == test_y)
        acc_matrix[s - 1, block] = acc
        #print(test_y)
        #print(test_y.shape)
        #print(YPred)
        #print(YPred.shape)
        # print(type(all_conf_matrix))
        all_conf_matrix += tf.math.confusion_matrix(tf.cast(test_y, tf.int32), tf.cast(np.argmax(YPred, axis=1), tf.int32))


    sv_name = f'confusion_mat_{block+1}.npy'
    np.save(sv_name, all_conf_matrix.numpy())  # TensorFlow tensor를 NumPy 배열로 변환하여 저장

    sv_name = 'acc_matrix.npy'
    np.save(sv_name, acc_matrix)

itr_matrix = itr(acc_matrix, totalcharacter, visual_cue + signal_length)

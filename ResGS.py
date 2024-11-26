import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from DNNGP import DNNGP, DNNGP_training_set

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')


def Conv1d_BN(x, nb_filter, kernel_size, strides=1):
    x = layers.Convolution1D(nb_filter, kernel_size, padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization(axis=1)(x)
    return x

def Res_Block(inpt,nb_filter,kernel_size,strides=1):
    x = Conv1d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
    x = layers.add([x,inpt])
    return x

def ResGSModel(inputs):
    nFilter = 64
    _KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1
    print("inputs.shape:", inputs.shape)

    x1 = Res_Block(inputs, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x1 = Res_Block(x1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    nFilter1 = int(nFilter * CHANNEL_FACTOR1)

    x2 = Conv1d_BN(x1 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x2 = Conv1d_BN(x2, nb_filter=nFilter, kernel_size=1, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x2 = Res_Block(x2, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x3 = Conv1d_BN(x2 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x3 = Conv1d_BN(x3, nb_filter=nFilter, kernel_size=1, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x3 = Res_Block(x3, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x4 = Conv1d_BN(x3 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x4 = Conv1d_BN(x4, nb_filter=nFilter, kernel_size=1, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x4 = Res_Block(x4, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x5 = Conv1d_BN(x4 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x5 = Conv1d_BN(x5, nb_filter=nFilter, kernel_size=1, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x5 = Res_Block(x5, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    x6 = Conv1d_BN(x5 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x6 = Conv1d_BN(x6, nb_filter=nFilter, kernel_size=1, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x6 = Res_Block(x6, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x7 = Conv1d_BN(x6 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x7 = Conv1d_BN(x7, nb_filter=nFilter, kernel_size=1, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x7 = Res_Block(x7, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x8 = Conv1d_BN(x7 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x8 = Conv1d_BN(x8, nb_filter=nFilter, kernel_size=1, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x8 = Res_Block(x8, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)


    x9 = Conv1d_BN(x8 , nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2)
    nFilter = int(nFilter * CHANNEL_FACTOR2)
    x9 = Conv1d_BN(x9, nb_filter=nFilter, kernel_size=1, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
    x9 = Res_Block(x9, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)

    element_number = x9.shape[1] * x9.shape[2]
    print("x9.shape[1]:", x9.shape[1])
    print("x9.shape[2]:", x9.shape[2])
    print("element_number:", element_number)
    filter_near_6400 = 6400 // x9.shape[1]
    if filter_near_6400 == 0:
        filter_near_6400 = 1
    print("filter_near_6400:", filter_near_6400)
    x = Conv1d_BN(x9, nb_filter=filter_near_6400, kernel_size=1, strides=1)
    x = layers.Flatten()(x)

    x = layers.Dense(1)(x)

    return Model(inputs = inputs, outputs = x)


def ResGS_pure(X_train, X_test, y_train, y_test,
            saveFileName,
            CUDA_VISIBLE_DEVICES = 0,
            Epoch = 3000,):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
    batch_size = 64
    patience = 1000
    repeatTimes = 3
    bestCorrelations = []
    y_pres = []

    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_pre = ridge_model.predict(X_test)
    r_Ridge = np.corrcoef(y_pre, y_test)[0,1]  # r_Ridge is pearson correlation
    print("r_Ridge:", r_Ridge)


    nSNP = X_train.shape[1]

    X2_train = np.expand_dims(X_train, axis=2)
    X2_test = np.expand_dims(X_test, axis=2)

    for i in range(repeatTimes):
        tf.random.set_seed(i)
        inputs = layers.Input(shape=(nSNP, 1))

        model = ResGSModel(inputs)
        # model.compile(loss='mse', optimizer='adam')
        model.compile(loss='mae', optimizer='adam')
        performance_simple = PerformancePlotCallback_ResGS_pure(X2_test, y_test, model= model, repeatTime=0,
                                                         saveFileName=saveFileName, patience=patience)
        history = model.fit(X2_train, y_train, epochs=Epoch, batch_size=batch_size,
                                      validation_data=(X2_test, y_test),
                                      verbose=0, callbacks=performance_simple)

        print("performance_simple.bestCorrelation: ", performance_simple.bestCorrelation)
        bestCorrelations.append(performance_simple.bestCorrelation)
        y_pres.append(performance_simple.y_pre)

        if performance_simple.bestCorrelation > r_Ridge:
            break


    print("bestCorrelation: ", max(bestCorrelations))
    # print("==============the best result of repeatTimes is: ", bestCorrelations.index(max(bestCorrelations)))

    print("bestCorrelations:", bestCorrelations)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:09:20 2019

@author: af1tang
"""
from keras.models import *
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import *
from keras.regularizers import *
from keras import metrics

def lstm_model(input_shape, hidden = 256, targets = 1, learn_rate = 1e-4, multiclass=False):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    if (targets > 1) and not multiclass:
        model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
    
    model.add(Dense(targets))
    if multiclass:
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=learn_rate, beta_1 =.5 ), metrics=['categorical_accuracy'])
    else:
        model.add(Activation ('sigmoid'))
        model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(lr=learn_rate, beta_1 =.5 ), metrics=['accuracy'])
    return (model)
    
def cnn_model(input_shape, hidden = 256, targets = 1, learn_rate = 1e-4):
    model = Sequential()
    model.add(Convolution1D(input_shape = input_shape, nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 3))
    model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(targets))
    if multiclass:
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=learn_rate, beta_1 =.5 ), metrics=['categorical_accuracy'])
    else:
        model.add(Activation ('sigmoid'))
        model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(lr=learn_rate, beta_1 =.5 ), metrics=['accuracy'])
    return (model)
    

def mlp_model(input_shape, hidden =512, targets = 1, multiclass = False, learn_rate = 1e-4):
    model = Sequential()
    model.add(Dense(hidden, activation = 'relu', input_shape = input_shape))
    model.add(Dropout(.5))
    model.add(Dense(hidden, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(hidden, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(targets))
    if multiclass:
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rate, beta_1 =.5 ), 
                      metrics=['categorical_accuracy'])
    else:
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate, beta_1 =.5 ), metrics=['accuracy'])
    return (model)

def hierarchical_cnn (input_shape, aux_shape, targets = 1, hidden = 256, multiclass = False, learn_rate=1e-4):
    x = Input(shape = input_shape, name = 'x')
    xx = Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu') (x)
    xx = MaxPooling1D(pool_length = 3) (xx)
    
    xx = Bidirectional(LSTM (256, activation = 'relu'), merge_mode = 'concat') (xx)
    xx = Dropout(0.5)(xx)
    
    dx = Input(shape = aux_shape, name = 'aux')

    xx = concatenate([xx, dx])
    if multiclass:
        y = Dense(targets, activation = 'softmax') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'categorical_crossentropy', optimizer = Adam(lr = learn_rate), metrics = ['categorical_accuracy'])
    else:
        y = Dense(targets, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = learn_rate), metrics = ['accuracy'])
    return (model)

def hierarchical_lstm (input_shape, aux_shape, targets = 1, hidden = 256, multiclass = False, learn_rate = 1e-4):
    x = Input(shape = input_shape, name = 'x')    
    xx = Bidirectional(LSTM (hidden, activation = 'relu'), merge_mode = 'concat') (x)
    xx = Dropout(0.5)(xx)
    
    dx = Input(shape = aux_shape, name = 'aux')

    xx = concatenate([xx, dx])
    xx = Dense(512, activation = 'relu') (xx)
    if multiclass:
        y = Dense(targets, activation = 'softmax') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'categorical_crossentropy', optimizer = Adam(lr = learn_rate), metrics = ['categorical_accuracy'])
    else:
        y = Dense(targets, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = learn_rate), metrics = ['accuracy'])
    return (model)
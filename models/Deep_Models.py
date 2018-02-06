# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:32:31 2017

@author: andy
"""
from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Activation, Dropout, TimeDistributed, Bidirectional, Masking, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.preprocessing import sequence
from keras import metrics

class lstm_trainer:
    def __init__(self, hierarchal = False, multiclass = False, input_shape = None, targets = 1, embed_shape = None, 
                 stateful = True, multi_targets=False, target_rep = False, embed = False):
        self.hierarchal = False
        self.multiclass = multiclass
        self.input_shape = input_shape
        self.targets = targets
        self.embed_shape = embed_shape
        self.multi_targets = multi_targets
        self.target_rep = target_rep
        self.embed =embed
        self.stateful = stateful
        self.model = self.lstm_train(self.input_shape, self.targets, self.embed_shape, self.stateful, self.multi_targets,
                                   self.target_rep, self.embed)
    
    def lstm_train(self, input_shape, targets = 1, embed_shape = None, stateful = False, multi_targets=False, 
                   target_rep = False, embed = False):
        model = Sequential()
        if embed:
            model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = True))
        else:
            model.add(Masking(mask_value=0, input_shape=input_shape))
            
        if (stateful == True):
            model.add(Bidirectional(LSTM(512, return_sequences = multi_targets, stateful =stateful), merge_mode = 'concat', batch_input_shape = (1, input_shape[0], input_shape[1])))
        else:
            model.add(Bidirectional(LSTM(512, return_sequences = multi_targets, stateful = stateful), merge_mode = 'concat'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
    
        if multi_targets:
            model.add(Bidirectional(LSTM(512, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
        
        if target_rep:
            model.add(TimeDistributed(Dense(targets, activation = 'softmax')))
        else:
            model.add(Dense(targets, activation = 'softmax'))
        optimizer = Adam(lr=1e-4, beta_1 =.5 )
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return (model)
    
        
    def multiclass_lstm(self, input_shape, embed_shape = None, embed = False, multi_targets = False):
        model = Sequential()
        if embed:
            model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = True))
        else:
            model.add(Masking(mask_value=0., input_shape=input_shape))
            
        model.add(Bidirectional(LSTM(512, return_sequences = multi_targets), merge_mode = 'concat'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
    
        model.add(Dense(8, activation = 'softmax'))
        optimizer = Adam(lr=1e-4, beta_1 =.5 )
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
        return (model)
    
    def hierarchal_lstm (self, input_shape, embed_shape, stateful= False, target_rep = False, multi_target = False, embed = False, multiclass = False):
        x = Input(shape = input_shape, name = 'x')
        xx = Masking(mask_value=0.)(x)
        
        xx = Bidirectional(LSTM (512, return_sequences = target_rep, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
        xx = Dropout(0.5)(xx)
        #if multi:
        #    xx = Bidirectional(LSTM (256, return_sequences = target_rep, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
        #    xx = Dropout(0.5)(xx)
        
        dx = Input(shape = embed_shape, name = 'dx')
    
        xx = concatenate([xx, dx])
        xx = Dense(512, activation = 'relu') (xx)
        if multiclass:
            y = Dense(8, activation = 'softmax') (xx)
            model = Model(inputs = [x, dx], outputs = [y])
            model.compile (loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['categorical_accuracy'])
        elif multi_target:
            y = Dense(25, activation = 'sigmoid') (xx)
            model = Model(inputs = [x, dx], outputs = [y])
            model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
        else:
            y = Dense(1, activation = 'sigmoid') (xx)
            model = Model(inputs = [x, dx], outputs = [y])
            model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
        return (model)
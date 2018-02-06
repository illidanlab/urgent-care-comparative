# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:40:31 2017

@author: andy
"""

import sys
import pandas as pd
import numpy as np
import pickle
import math, random
import datetime, time
import tensorflow as tf
import gensim
import matplotlib.pyplot as plt

import argparse

#from pandas.tools.plotting import scatter_matrix
from scipy import stats
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from scipy.stats import skew, ttest_ind
from itertools import combinations
from time import sleep
from datetime import timedelta
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV as random_search

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
from sklearn.utils import shuffle, class_weight

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

def main(xs, x_names, dxs, y, w2vs, maxlens, multi, mode, epochs):
    
    data  = {}   
    multiclass = input("Multiclass Settings? [Y/N] ")
    model_file = input("Model File Location: ")
    
    for xi in range(len(xs)):
        x_name = x_names[xi]
        x = xs[xi]
        try:
            dx = dxs[xi]
        except:
            dx = None
        maxlen = maxlens[xi]
        w2v = w2vs[xi]
        
        if w2v:
            if maxlen<1000:
                x = [[i+1 for i in d] for d in x]
            x = sequence.pad_sequences(x, maxlen, value = 0)
            input_shape = (x.shape[1],)
            vocab_length = max([max(k) for k in x])
            embed_shape = (vocab_length +1 , 200, maxlen)
        else:
            input_shape = (x.shape[1], x.shape[2])
            if dx is not None:
                embed_shape = (dx.shape[1],)
            else:
                embed_shape = None
        
        params = {'input_shape':input_shape, 'embed_shape': embed_shape, 'multi': multi, 'w2v':w2v}

        print (params.items())
        print(y.shape)
        dat, model = run_experiment(x =x, dx = dx, y =y , params = params, epochs = epochs, mode = mode, multiclass = multiclass)
        #dat, model = make_model(x=x, dx= dx, y=y, params= params, epochs = epochs, mode = mode, multiclass = multiclass)
        data[x_name] = dat
        model.save(model_file+x_name+'.h5')
        
    return (data)

def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)
    
def W2V (dx):
    dx = np.ndarray.tolist(dx)
    SG = gensim.models.Word2Vec(sentences = dx, sg = 1, size = 200, window = 5, min_count = 0, hs = 1, negative = 0)
    weights = SG.wv.syn0
    vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
    w2i, i2w = vocab_index(vocab)
    
    #turn sentences into word vectors for each admission
    dx = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv)) for vv in dx]    
    #word vectors here
    w2v = [] 
    
    for sentence in dx:
        one_hot = np.zeros((len(sentence), weights.shape[0]))
        one_hot[np.arange(len(sentence)), sentence] = 1
        one_hot = np.sum(one_hot, axis= 0)
        w2v.append(np.dot(one_hot.reshape(one_hot.shape[0]), weights))
    return (w2v, SG, weights, vocab)
        
def run_experiment(x, dx, y, params, epochs = 30, mode = 'lstm', multiclass = 'N'):
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    data = {}

    start = time.time(); count = 0
    
    if params['multi'] == True:
        if multiclass != 'Y': 
            tmp = y[:,-1]
        else:
            tmp = y
    else:
        tmp = y
        
    for train_index, test_index in skf.split(x, tmp):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = [] 
        
        X_train, X_test = x[train_index], x[test_index]
        if dx is not None:
            dx_train, dx_test = dx[train_index], dx[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if (params['multi']) and (multiclass != 'Y'):
            if dx is not None:
                xs, ds, ys = X_train, dx_train, y_train
            else:
                xs, ys = X_train, y_train
        else:
            if dx is not None:
                xs, ds, ys = hierarchal_subsample(X_train, dx_train, y_train, 1.0)
            else:
                xs, ys = balanced_subsample(X_train, y_train, 1.0)
            ys = np.array([[i] for i in ys])
        
        if multiclass == 'Y':
            lb = LabelBinarizer()      
            ys = lb.fit_transform(ys)
            y_test = lb.fit_transform(y_test)
        
        model = None
        if mode == 'lstm':
            model = lstm_train(**params)
        elif mode == 'cnn':
            model = cnn_train(**params)
        elif mode == 'mlp':
            model = mlp_train(**params)
        elif mode == 'hierarchal_lstm':
            if multiclass == 'Y':
                multiclass = True
            else:
                multiclass = False
            params['multiclass'] = multiclass
            model = hierarchal_lstm(**params)
            multiclass = 'Y'
        elif mode == 'hierarchal_cnn':
            if multiclass == 'Y':
                multiclass = True
            else:
                multiclass = False
            params['multiclass'] = multiclass
            model = hierarchal_cnn(**params)
            multiclass = 'Y'
        elif mode == 'multiclass_lstm':
            model = multiclass_lstm(**params)
        elif mode == 'multiclass_cnn':
            model = multiclass_cnn(**params)
        else:
            print("ERROR IN MODE SELECTION.")
            return;
        
        if dx is not None:
            model.fit(x = [xs, ds], y= ys, epochs = epochs)
            y_pred = model.predict(x = [xs, ds])
            yhat = model.predict(x= [X_test, dx_test])
        else:
            model.fit(xs, ys, epochs = epochs)
            y_pred = model.predict(xs)
            yhat = model.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        tr_roc_auc = dict()
        f1= dict()
        te_roc_auc = dict()
        te_matrix = dict()
        
        if (params['multi'] == True):
            for idx in range(ys[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(ys[:, idx], y_pred[:, idx])
                tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                
            fpr["micro"], tpr["micro"], _ = roc_curve(ys.ravel(), y_pred.ravel())                
            tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   
            
            for idx in range(ys[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], yhat[:, idx])
                te_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                te_matrix[idx] = confusion_matrix(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                f1[idx] = f1_score(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yhat.ravel())
            te_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            f1["micro"] = f1_score(y_test.ravel(), np.array([round(i) for i in yhat.ravel()]))
        else:
            fpr, tpr, _ = roc_curve(ys, y_pred)
            tr_roc_auc = auc(fpr, tpr)
            te_matrix = confusion_matrix(y_test, np.array([round(i[0]) for i in yhat]))
            f1 = f1_score(y_test, np.array([round(i[0]) for i in yhat]))
            
            fpr, tpr, _ = roc_curve(y_test, yhat)
            te_roc_auc = auc(fpr, tpr)
        
        data[count]['tr_auc'].append(tr_roc_auc)
        data[count]['f1_score'].append(f1)
        data[count]['te_auc'].append(te_roc_auc)
        data[count]['te_matrix'].append(te_matrix)
    
    return (data, model)
    
def make_model(x, dx, y, params, epochs = 30, mode = 'lstm', multiclass = 'N'):
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    data = {}
    lb = LabelBinarizer()
    #mlb = MultiLabelBinarizer()
    
    train_index, test_index = list(skf.split(x, y))[0]
    data = {}
    data['tr_auc'] = []
    data['f1_score'] = []
    data['te_auc'] = []
    data['te_matrix'] = []
    X_train, X_test = x[train_index], x[test_index]
    if dx is not None:
        dx_train, dx_test = dx[train_index], dx[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if len(y.shape) ==1:
        if dx is not None:
            xs, ds, ys = hierarchal_subsample(X_train, dx_train, y_train, 1.0)
        else:
            xs, ys = balanced_subsample(X_train, y_train, 1.0)
            
        if params['multi'] == True:
            ys = lb.fit_transform(ys)
            y_test = lb.fit_transform(y_test)
        else:
            ys = np.array([[i] for i in ys])
            
    else:
        if dx is not None:
            ds = dx_train
        xs, ys = X_train, y_train
    #sample_wt = None

    model = None
    if mode == 'lstm':
        model = lstm_train(**params)
    elif mode == 'cnn':
        model = cnn_train(**params)
    elif mode == 'mlp':
        model = mlp_train(**params)
    elif mode == 'hierarchal_lstm':
        if multiclass == 'Y':
            multiclass = True
        else:
            multiclass = False
        params['multiclass'] = multiclass
        model = hierarchal_lstm(**params)
    elif mode == 'hierarchal_cnn':
        if multiclass == 'Y':
            multiclass = True
        else:
            multiclass = False
        params['multiclass'] = multiclass
        model = hierarchal_cnn(**params)
    elif mode == 'multiclass_lstm':
        model = multiclass_lstm(**params)
    elif mode == 'multiclass_cnn':
        model = multiclass_cnn(**params)
    else:
        print("ERROR IN MODE SELECTION.")
        return;

    if dx is not None:
        model.fit(x = [xs, ds], y= ys, epochs = epochs)
        y_pred = model.predict(x = [xs, ds])
        yhat = model.predict(x= [X_test, dx_test])
    else:
        model.fit(xs, ys, epochs = epochs)
        y_pred = model.predict(xs)
        yhat = model.predict(X_test)
    fpr = dict()
    tpr = dict()
    tr_roc_auc = dict()
    f1= dict()
    te_roc_auc = dict()
    te_matrix = dict()
    if (params['multi'] == True):
        for idx in range(ys[0].shape[0]):
            fpr[idx], tpr[idx], _ = roc_curve(ys[:, idx], y_pred[:, idx])
            tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
        fpr["micro"], tpr["micro"], _ = roc_curve(ys.ravel(), y_pred.ravel())                
        tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   
        for idx in range(ys[0].shape[0]):
            fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], yhat[:, idx])
            te_roc_auc[idx] = auc(fpr[idx], tpr[idx])
            te_matrix[idx] = confusion_matrix(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
            f1[idx] = f1_score(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yhat.ravel())
        te_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        f1["micro"] = f1_score(y_test.ravel(), np.array([round(i) for i in yhat.ravel()]))
    else:
        fpr, tpr, _ = roc_curve(ys, y_pred)
        tr_roc_auc = auc(fpr, tpr)
        te_matrix = confusion_matrix(y_test, np.array([round(i[0]) for i in yhat]))
        f1 = f1_score(y_test, np.array([round(i[0]) for i in yhat]))

        fpr, tpr, _ = roc_curve(y_test, yhat)
        te_roc_auc = auc(fpr, tpr)
    data['tr_auc'].append(tr_roc_auc)
    data['f1_score'].append(f1)
    data['te_auc'].append(te_roc_auc)
    data['te_matrix'].append(te_matrix)

    return (data, model)

    
def r_search(x, y, input_shape):
    #random search params
    mlp_params = {'units': [64, 128, 256, 512], 'rate': sp_rand(.2, .9)}
    lstm_params = {'units': [64, 128, 256, 512], 'rate': sp_rand(.2, .9)}
    cnn_params = {'filters': [32, 64, 128, 256, 512], 'filter_length': [2, 3, 4, 5, 6], 'pool_size': [2, 3]}
    
    data = {}
    xs, ys = balanced_subsample(x, y)
    lst = [mlp_train(input_shape), lstm_train(input_shape), cnn_train(input_shape)]
    names = ['MLP', 'LSTM', 'CNN']
    params = [mlp_params, lstm_params, cnn_params]
    for idx in range(len(lst)):
        n_iter_search = 60
        start = time.time()    
        rsearch = random_search(estimator = lst[idx], param_distributions = params[idx], n_iter=n_iter_search, scoring='roc_auc', fit_params=None, n_jobs=1, iid=True, refit=True, cv=3, verbose=10, random_state=8)
        rsearch.fit(xs, ys)
        data[names[idx]] = rsearch.cv_results_
        print (names[idx]+" results complete.")
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time.time() - start), n_iter_search))
    return (data)
            
    
def balanced_subsample(x, y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def hierarchal_subsample(x, dx, y,subsample_size=1.0):

    class_xs = []
    class_dxs = []
    min_elems_x = None
    min_elems_d = None

    for yi in np.unique(y):
        elems_x = x[(y == yi)]
        elems_d = dx[(y==yi)]
        class_xs.append((yi, elems_x))
        class_dxs.append((yi, elems_d))
        if min_elems_x == None or elems_x.shape[0] < min_elems_x:
            min_elems_x = elems_x.shape[0]
            min_elems_d = elems_d.shape[0]

    use_elems_x = min_elems_x
    use_elems_d = min_elems_d
    if subsample_size < 1:
        use_elems_x = int(min_elems_x*subsample_size)
        use_elems_d = int(min_elems_d*subsample_size)

    xs = []
    dxs = []
    ys = []

    for lst1, lst2 in zip(class_xs, class_dxs):
        ci = lst1[0]
        this_xs = lst1[1]
        this_dxs = lst2[1]
        
        if len(this_xs) > use_elems_x:
            this_xs, this_dxs = shuffle(this_xs, this_dxs)

        x_ = this_xs[:use_elems_x]
        d_ = this_dxs[:use_elems_d]
        y_ = np.empty(use_elems_x)
        y_.fill(ci)

        xs.append(x_)
        dxs.append(d_)
        ys.append(y_)

    xs = np.concatenate(xs)
    dxs = np.concatenate(dxs)
    ys = np.concatenate(ys)

    return xs, dxs, ys
    
def lstm_train(input_shape, embed_shape = None, stateful = False, target_rep = False, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = True))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        
    if (stateful == True):
        model.add(Bidirectional(LSTM(256, return_sequences = multi, stateful =stateful), merge_mode = 'concat', batch_input_shape = (1, input_shape[0], input_shape[1])))
    else:
        model.add(Bidirectional(LSTM(256, return_sequences = multi, stateful = stateful), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    if multi:
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
    
    if (target_rep == True):
        if multi:
            model.add(TimeDistributed(Dense(25, activation = 'sigmoid')))
        else:
            model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        if multi:
            model.add(Dense(25, activation = 'sigmoid'))
        else:
            model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)
    
def cnn_train(input_shape, embed_shape = None, stateful = False, target_rep = False, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = False))
        model.add(Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    else:
        model.add(Convolution1D(input_shape = input_shape, nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 3))
    
    model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    
    #model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    if (target_rep == True):
        if multi:
            model.add(TimeDistributed(Dense(25, activation = 'sigmoid')))
        else:
            model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        if multi:
            model.add(Dense(25, activation = 'sigmoid'))
        else:
            model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)
    
def multiclass_lstm(input_shape, embed_shape = None, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = True))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        
    model.add(Bidirectional(LSTM(256, return_sequences = False), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(8, activation = 'softmax'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    return (model)

def multiclass_cnn(input_shape, embed_shape = None, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = False))
        model.add(Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    else:
        model.add(Convolution1D(input_shape = input_shape, nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 3))
        
    model.add(Bidirectional(LSTM(256, return_sequences = False), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(8, activation = 'softmax'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    return (model)

def mlp_train(input_shape, multi = False):
    model = Sequential()
    model.add(Dense(512, activation = 'relu', input_shape = input_shape))
    model.add(Dropout(.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(.5))

    if multi:
        model.add(Dense(25, activation = 'sigmoid'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)

def hierarchal_cnn (input_shape, embed_shape, stateful= False, target_rep = False, multi = False, w2v= False, multiclass = False):
    x = Input(shape = input_shape, name = 'x')
    
    xx = Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu') (x)
    xx = MaxPooling1D(pool_length = 3) (xx)
    
    xx = Bidirectional(LSTM (256, return_sequences = False, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
    xx = Dropout(0.5)(xx)
    
    dx = Input(shape = embed_shape, name = 'dx')

    xx = concatenate([xx, dx])
    #xx = Dense(256, activation = 'relu') (xx)
    if multiclass:
        y = Dense(8, activation = 'softmax') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['categorical_accuracy'])
    elif multi:
        y = Dense(25, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    else:
        y = Dense(1, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)

def hierarchal_lstm (input_shape, embed_shape, stateful= False, target_rep = False, multi = False, w2v = False, multiclass = False):
    x = Input(shape = input_shape, name = 'x')
    xx = Masking(mask_value=0.)(x)
    
    xx = Bidirectional(LSTM (256, return_sequences = target_rep, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
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
    elif multi:
        y = Dense(25, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    else:
        y = Dense(1, activation = 'sigmoid') (xx)
        model = Model(inputs = [x, dx], outputs = [y])
        model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)

def ttest(data):
    df1 = pd.DataFrame(data, columns = list(data.keys()))
    df2 = pd.DataFrame(data, columns = list(data.keys()))
    import numpy as np
    mat = np.zeros((df1.shape[1], df2.shape[1]))
    mat
    for i in range(df1.shape[1]):
        for j in range(df2.shape[1]):
            t,p = ttest_ind(df1[df1.columns[i]], df2[df2.columns[j]])
            mat[i,j] = p
    dfmat = pd.DataFrame(mat, columns = df2.columns, index = df1.columns)
    return(dfmat)
    
def reporting (dat, task, m, f, dct, multi = False):
    data = {}
    #pheno25
    #task = '25 ddx'
    data[task][m+'_'+f] = {}
    
    if multi:
        f1 = []; auc = []
        sen = {}
        spec = {}
        for loop in list(dat[f].keys()):
            sen[loop]= []
            spec[loop] = []
            f1.append(list(dat[f][loop]['f1_score'][0].values()))
            auc.append(list(dat[f][loop]['te_auc'][0].values()))
            for mat in list(dat[f][loop]['te_matrix'][0].values()):
                tn, fp, fn, tp = mat.ravel()
                sen[loop].append(1.0* (tp/(tp+fn)))
                spec[loop].append(1.0* (tn/(tn+fp)))
        f1 = (np.mean(f1, axis = 0), np.std(f1, axis = 0))
        auc = (np.mean(auc, axis =0), np.std(auc, axis = 0))
        sen = (np.mean(list(sen.values()), axis =0), np.std(list(sen.values()), axis = 0))
        spec = (np.mean(list(spec.values()), axis =0), np.std(list(spec.values()), axis = 0))
        
        for n in range(len(f1[0])):
            data[task][m+'_'+f]['f1_'+dct[n]] = '{0:.3}'.format(f1[0][n]) + '({0:.3})'.format(f1[1][n])
            data[task][m+'_'+f]['auc_'+dct[n]] = '{0:.3}'.format(auc[0][n]) + '({0:.3})'.format(auc[1][n])
            try:
                data[task][m+'_'+f]['sen_' + dct[n]] = '{0:.3}'.format(sen[0][n]) + '({0:.3})'.format(sen[1][n])
                data[task][m+'_'+f]['spec_' + dct[n]] = '{0:.3}'.format(spec[0][n])+ '({0:.3})'.format(spec[1][n])
            except:pass
    else:
        f1 = []; auc = []
        sen = []
        spec = []
        for loop in list(dat[f].keys()):
            f1.append(dat[f][loop]['f1_score'][0])
            auc.append(dat[f][loop]['te_auc'][0])
            tn, fp, fn, tp = dat[f][loop]['te_matrix'][0].ravel()
            sen.append(1.0* (tp/(tp+fn)))
            spec.append(1.0* (tn/(tn+fp)))
        f1 = (np.mean(f1, axis = 0), np.std(f1, axis = 0))
        auc = (np.mean(auc, axis =0), np.std(auc, axis = 0))
        sen = (np.mean(sen, axis =0), np.std(sen, axis =0))
        spec = (np.mean(spec, axis =0), np.std(spec, axis=0))
        
        data[task][m+'_'+f]['f1'] = '{0:.3}'.format(f1[0]) + ' ({0:.3})'.format(f1[1])
        data[task][m+'_'+f]['auc'] = '{0:.3}'.format(auc[0]) + ' ({0:.3})'.format(auc[1])
        data[task][m+'_'+f]['sen'] = '{0:.3}'.format(sen[0]) + ' ({0:.3})'.format(sen[1])
        data[task][m+'_'+f]['spec'] = '{0:.3}'.format(spec[0]) + ' ({0:.3})'.format(spec[1])
    
    return data

def str2bool(v):
    return v.lower() in ("True", "true", "yes", "Yes", 't', 'T', '1', 'YES', 'Y', 'y')
    
if __name__ == '__main__':
    xs = []; dxs = []; maxlens = []
    
    parser = argparse.ArgumentParser("Input X, Y, and settings.")
    parser.add_argument('--x', type = str, help = "Enter X's.")
    parser.add_argument('--x_name', type = str, action = 'append', help = "Enter X names.")
    parser.add_argument('--dx', default = None, help = "Enter Auxiliary Inputs.")
    parser.add_argument('--y', type = str, help = 'Enter task w/ labels Y.')
    parser.add_argument('--w2v', type = str2bool, action = 'append', help = "Do you want word2vec embeddings on input?")
    parser.add_argument('--maxlen', default =None, action = 'append', help = "Enter Padding length.")
    parser.add_argument('--multi', action = 'store_true', help = "Multilabel classification?")
    parser.add_argument('--mode', choices = ['lstm', 'hierarchal_lstm', 'cnn', 'hierarchal_cnn', 'multiclass_lstm', 'multiclass_cnn'])
    parser.add_argument('--epochs', type = int, default = 30, help = "Enter number of epochs to train model on.")
    #parser.add_argument('--o', help = "Enter Output File name.")
    args = parser.parse_args("--x /home/andy/Desktop/MIMIC/vars/npy/Xs/s12_Xs.pkl  --x_name 12ts --x_name dix --y /home/andy/Desktop/MIMIC/vars/npy/Ys/Ylos_multinomial.npy --w2v False --w2v True --maxlen None --maxlen 39 --mode multiclass_cnn --multi --epochs 100".split())
    #args = parser.parse_args()

    try:
        with open(args.x, 'rb') as f:
            xs = pickle.load(f)
    except:
        x = np.load(args.x)
        xs = [x]
    
    try:
        dx = np.load(args.dx)
        dxs = [dx]
    except: dxs = None
    
    for m in args.maxlen:
        try: maxlens.append(int(m))
        except: maxlens.append(None)
        
    y = np.load(args.y)       
    data = main(xs = xs, x_names = args.x_name, dxs = dxs, y= y, w2vs = args.w2v, maxlens = maxlens, multi = args.multi, mode = args.mode, epochs = args.epochs)    

    #o = '/home/andy/Desktop/MIMIC/dat/CNN_results/cnn_los/df.pkl'    
    #o = '/home/andy/Desktop/MIMIC/dat/LSTM_results/lstm_los/df.pkl'
    
    try:
        with open(args.o, 'wb') as f:
            pickle.dump(data, f)
    except: print(data)
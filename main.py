#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:32:17 2019

@author: af1tang
"""
import os
import numpy as np
import utilities
import argparse
import pickle
from deep_models import *

from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import roc_curve, auc as auc_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle


def pipeline(X, Z, opts):
    from keras.callbacks import ModelCheckpoint
    skf = StratifiedKFold(n_splits=5)
    data = {}
    count = 0
    
    y = get_task(opts)
    targets, multiclass, deep = get_setup(opts)
    if targets>1:
        ref_target = y[:,-1]
        for idx in range(len(y[0,:])):
            y[:, idx][y[:,idx]>1] = 1
    else:
        ref_target = y
    
    X,y= np.array(X), np.array(y)
    if Z is not None: Z = np.array(Z)
    for train_index, test_index in skf.split(X, ref_target):
        count +=1
        print ("KFold #{0}".format(count))
        
        X_tr, X_te = X[train_index], X[test_index]
        if Z is not None:
            Z_tr, Z_te = Z[train_index], Z[test_index]
            if (not deep) or (opts.model =='mlp'):
                X_tr, X_te = np.append(X_tr, Z_tr, axis=1), np.append(X_te, Z_te, axis=1)
        y_tr, y_te = y[train_index], y[test_index]
        
        if len(X_tr.shape) >2:
            input_shape = (X_tr.shape[1], X_tr.shape[-1])
        else:
            input_shape= (X_tr.shape[-1],)
        if Z is not None: aux_shape = (Z.shape[-1],)
        else: aux_shape = None
        #make model 
        filename = os.path.join(opts.checkpoint_path, 'best_model' )
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        model = make_model(opts, input_shape, aux_shape)
        
        #train over epochs
        for e in range(opts.nepochs):
            #subsample
            if targets>1:
                if Z is not None:
                    xs, zs, ys = shuffle(X_tr, Z_tr, y_tr)
                else:
                    xs, ys = shuffle(X_tr, y_tr)
            else:
                if (Z is not None) and deep:
                    xs, zs, ys = hierarchical_subsample(X_tr, Z_tr, y_tr, 1.0)
                else:
                    xs, ys = balanced_subsample(X_tr, y_tr, 1.0)
                ys = np.array([[i] for i in ys])

            if deep:
                if (Z is not None) and (opts.model !='mlp'):
                    model.fit(x = [xs, zs], y= ys,
                              batch_size = opts.batch_size, callbacks=[checkpoint],
                              validation_split=.2, epochs = 1, verbose=2)
                else:
                    model.fit(x = xs, y=ys, batch_size = opts.batch_size, callbacks=[checkpoint],
                              validation_split=.2, epochs=1, verbose=2)
            else:
                model = model.fit(xs, ys)
                
        if deep: model.load_weights(filename)
        else: 
            with open(filename, 'wb') as f: pickle.dump(model, f)
        if (Z is not None) and deep and (opts.model !='mlp'):
            tr_auc, _, _, _ = test_model(x = [X_tr, Z_tr], y= y_tr, model = model, n_classes=targets)
            te_auc, f1_score, sen, spec = test_model(x = [X_te, Z_te], y= y_te, 
                                                     model = model, n_classes=targets)
        else:
            tr_auc, _, _, _= test_model(x = X_tr, y= y_tr,model = model, n_classes=targets)
            te_auc, f1_score, sen, spec = test_model(x = X_te, y= y_te, model = model, n_classes = targets)
            
        data[count] = {'tr_auc': tr_auc, 'f1_score':f1_score, 'te_auc': te_auc, 'sen':sen, 'spec': spec}
    
    return model, data

### Scoring ###
def test_model(x,y, model, n_classes):
    yhat = model.predict(x)
    if n_classes >1:
        roc_auc, f1, sen, spec = multi_score(y, yhat)
    else:
        roc_auc, f1, sen, spec= single_score(y, yhat)
    return roc_auc, f1, sen, spec

def single_score(y_te, yhat):
    fpr, tpr, thresholds = roc_curve(y_te, yhat)
    roc_auc = auc_score(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
    yhat=[int(i) for i in yhat]
    #matrix = confusion_matrix(y_te, yhat)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
    sen=1.0* (tp/(tp+fn))
    spec=1.0* (tn/(tn+fp))
    f1=f1_score(y_te,yhat)
    return roc_auc, f1, sen, spec

def micro_score(y_te, yhat):
    yhat, y_te = np.array(yhat), np.array(y_te)
    yhat, y_te = yhat.ravel(), y_te.ravel()
    fpr, tpr, thresholds = roc_curve(y_te, yhat)
    roc_auc = auc_score(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
    yhat=[int(i) for i in yhat]
    f1=f1_score(y_te,yhat)
    return roc_auc, f1

def multi_score(y_te, ypred):
    ypred, y_te = np.array(ypred), np.array(y_te)
    sens, specs, aucs, f1s= {}, {},{}, {}
    for idx in range(ypred.shape[1]):
        y_true, yhat = y_te[:, idx], ypred[:, idx]
        fpr, tpr, thresholds = roc_curve(y_true, yhat)
        aucs[idx] = auc_score(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        thresh = thresholds[optimal_idx]
        yhat[yhat>=thresh] = 1
        yhat[yhat<thresh] = 0
        f1s[idx] = f1_score(y_true, yhat)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        sens[idx]=1.0* (tp/(tp+fn))
        specs[idx]=1.0* (tn/(tn+fp))
    aucs['micro'], f1s['micro'] = micro_score(y_te, ypred)
    return aucs, f1s, sens, specs

### Main ###
def main(opts):
    features = np.load(opts.features_dir)
    if opts.auxiliary_dir:
        auxiliary = np.load(opts.auxiliary_dir)
    else:
        auxiliary = None

    model, stats = pipeline(X=features, Z=auxiliary, opts = opts)
    return model, stats


def get_task(opts):
    with open(opts.y_dir, 'rb') as f:
        labels = pickle.load(f)
    dct = {'mort':0, 'readmit': 1, 'los': 2, 'dx':3 }
    task = [yy[dct[opts.task]] for yy in labels]
    return np.array(task)

def get_setup(opts):
    multiclass= False
    if opts.task == 'los': multiclass=True; targets = 8
    elif opts.task == 'dx': targets=25
    else: targets=1
    
    if opts.model in ['lstm', 'cnn', 'mlp']: deep = True
    else: deep=False
    return targets, multiclass, deep

def take_names(opts):
    #x_name
    if 'X48' in opts.features_dir:
        x_name = 'x48'
    elif 'X' in opts.features_dir:
        x_name = 'x19'
    elif 'sentence' in opts.features_dir:
        x_name = 'sent'
    elif 'onehot' in opts.features_dir:
        x_name = 'ohv'
    else:
        x_name = None
    #aux_name
    if opts.auxiliary_dir:
        if 'w2v' in opts.auxiliary_dir:
            aux_name = 'w2v'
        elif 'h2v' in opts.auxiliary_dir:
            aux_name = 'h2v'
        elif 'demo' in opts.auxiliary_dir:
            aux_name = 'demo'
        else:
            aux_name='None'
    else:
        aux_name = 'None'    
    return x_name, aux_name

def make_model(opts, input_shape, aux_shape):
    targets, multiclass, deep = get_setup(opts)
        
    if opts.model == 'lstm':
        if aux_shape:
            model = hierarchical_lstm(input_shape=input_shape, aux_shape=aux_shape,
                                      targets=targets, hidden = opts.hidden_size, 
                                      multiclass= multiclass, learn_rate=opts.learning_rate)
        else:
            model = lstm_model(input_shape=input_shape, targets=targets, hidden = opts.hidden_size, 
                                      multiclass= multiclass, learn_rate=opts.learning_rate)
    elif opts.model == 'cnn':
        if aux_shape:
            model = hierarchical_cnn(input_shape=input_shape, aux_shape=aux_shape,
                                      targets=targets, hidden = opts.hidden_size, 
                                      multiclass= multiclass, learn_rate=opts.learning_rate)
        else:
            model = cnn_model(input_shape=input_shape, targets=targets, hidden = opts.hidden_size, 
                                      multiclass= multiclass, learn_rate=opts.learning_rate)
    elif opts.model == 'mlp':
        model = mlp_model(input_shape=input_shape, targets=targets, hidden = opts.hidden_size, 
                                      multiclass= multiclass, learn_rate=opts.learning_rate)
    elif opts.model == 'svm':
        if targets>1: model = OneVsRestClassifier(SVM(C=50000, kernel = 'rbf', max_iter= 1000, verbose = True, decision_function_shape = 'ovr', probability = True))
        else: model = SVM(C=1e4, kernel = 'linear', verbose = True, probability = True, max_iter= 1000)
    elif opts.model == 'rf':
        if targets>1: model = RF(n_estimators = 450, class_weight = 'balanced', criterion = 'entropy', bootstrap = False, verbose = 1)
        else: model = RF(n_estimators = 450, verbose = 1)
    elif opts.model =='gbc':
        if targets>1: model = OneVsRestClassifier(GBC(n_estimators = 484, learning_rate = 0.0984, verbose = 1))
        else: model = GBC(n_estimators = 400, learning_rate = 0.09, verbose = 1)
    elif opts.model == 'lr':
        if targets>1: model = OneVsRestClassifier(LR(max_iter= 1000, class_weight = 'balanced', multi_class = 'ovr', C = .09, penalty = 'l1', verbose = 1))    #sag if multiclass/multilabel
        else: model = LR(C = 1e-3, penalty = 'l2', verbose = 1)    #sag if multiclass/multilabel
    else:
        model = None
    return model

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--features_dir', type=str, default='/local_mimic/save/X19.npy',
                        help='Path to the uniform feature matrix (X19 or X48), or diagnostic history (sentences or onehot) file.')
    parser.add_argument('--auxiliary_dir', type=str, default='/local_mimic/save/w2v.npy',
                        help='Path to the auxiliary features (w2v, h2v or demo) file.')
    parser.add_argument('--y_dir', type=str, default='/local_mimic/save/y',
                        help='Path to the task labels (Y) file.')
    parser.add_argument('--model', default= 'lstm',
                        choices=['lstm' , 'cnn', 'mlp', 'svm', 'rf', 'lr', 'gbc'],
                        help='Type of model to use: lstm (default), cnn, mlp, svm, rf, lr, gbc.')
    parser.add_argument('--task', choices = ['readmit', 'mort', 'los','dx' ], default='mort',
                        help='Target task: readmission (readmit), mortality (mort), los, diagnosis (dx).')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The size of the hidden units (for deep models).')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='The max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='The learning rate (default 0.005)')

    parser.add_argument('--checkpoint_dir', type=str, default='/local_mimic/save/checkpoint',
                        help='Set the directry to store the best model checkpoints.')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
        else:
            print('{:>30}: {:<30}'.format(key, 'None').center(80))
    print('=' * 80)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)

    x_name, aux_name = take_names(opts)
    model_name = '{}-{}-{}-{}'.format(opts.model, opts.task, x_name, aux_name)
    opts.checkpoint_path = os.path.join(opts.checkpoint_dir, model_name)

    utilities.create_dir_if_not_exists(opts.checkpoint_path)
    utilities.create_dir_if_not_exists(os.path.join(opts.checkpoint_path, 'scores'))

    scores_folder = os.path.join(opts.checkpoint_path, 'scores')
    model, stats = main(opts)
    with open(scores_folder + '/raw_stats', 'wb') as f:
        pickle.dump(stats, f)
    
        
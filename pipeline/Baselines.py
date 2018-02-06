# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:20:56 2017

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
from scipy.stats import skew
from time import sleep
from datetime import timedelta
from sklearn.preprocessing import Imputer, MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV as random_search
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle, class_weight
from sklearn.neural_network import MLPClassifier as MLP


x_file = ''
y_file = ''

def main(xs, x_names, y, task):
    
    if task == 'multiclass':
        lr = OneVsRestClassifier(LR(max_iter= 1000, class_weight = 'balanced', multi_class = 'ovr', C = .09, penalty = 'l1', verbose = 1))    #sag if multiclass/multilabel
        svm = OneVsRestClassifier(SVM(C=50000, kernel = 'rbf', max_iter= 1000, verbose = True, decision_function_shape = 'ovr', probability = True))
        rf = RF(n_estimators = 450, class_weight = 'balanced', criterion = 'entropy', bootstrap = False, verbose = 1)
        gbc = OneVsRestClassifier(GBC(n_estimators = 484, learning_rate = 0.0984, verbose = 1))
        mlp = MLP(hidden_layer_sizes=(64, 64,), alpha = .006, verbose = True)
    else:
        lr = LR(C = 1e-3, penalty = 'l2', verbose = 1)    #sag if multiclass/multilabel
        svm = SVM(C=1e4, kernel = 'linear', verbose = True, probability = True, max_iter= 1000)
        rf = RF(n_estimators = 450, verbose = 1)
        gbc = GBC(n_estimators = 400, learning_rate = 0.09, verbose = 1)
        mlp = MLP(hidden_layer_sizes=(64, 64,), activation= 'relu', solver= 'adam', batch_size= 'auto', learning_rate= 'adaptive', learning_rate_init=1e-3, max_iter=500, shuffle=True, random_state=None, tol=0.0000001, verbose=True, early_stopping=True, validation_fraction=0.1)

    if (task == 'multilabel'):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y)

    fileout = input("Enter Output file name: ")
    models = [lr, svm, rf, gbc, mlp]
    names = ['LR', 'SVM', 'RF', 'GBC', 'MLP']
    data = {}    
    for idx in range(len(models)):
        data [names[idx]]= {}
        for xi in range(len(x_names)):
            x = xs[xi]
            x_name = x_names[xi]
            if task == 'multilabel':
                data[names[idx]][x_name] = {}
                for ix in range(25):
                    dat, model = run_experiment(x, y[:, ix], models[idx], task)
                    data[names[idx]][x_name][ix] = dat
                    with open(fileout+names[idx]+"_"+x_name+'.pkl', 'wb') as f:
                        pickle.dump(model, f, protocol=2)
            else:
                dat, model = run_experiment(x, y, models[idx], task)
                data[names[idx]][x_name] = dat
                with open(fileout+names[idx]+"_"+x_name+'.pkl', 'wb') as f:
                        pickle.dump(model, f, protocol=2)
        
    return (data)

def build_XY (x, y, option):
    
    #flatten X
    x = X_48hr (x)
    
    #w2v conversion
    w2v = W2V(dx)
    #standardize Word Vectors (optional)
    scaler = StandardScaler()
    w2v = scaler.fit_transform(w2v)
    #concatenate X with w2v
    #first, extend w2v to 3D tensor
    #w2v = np.repeat(w2v[:, np.newaxis, :], maxlen, axis=1)
    #make new X
    #x = np.concatenate((x, w2v), axis = 2)
    
    #mortality and readmission labels
    ym = y[:, 2]
    yr = y[:, -1]
    return (w2v, y)

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

def X_48hr (X):
    #standardize X
    tmp = []
    maxlen = []
    for i in range(len(X)):
        if len(X[i]) < 24: 
            pass
        else:
            tmp.append(np.array(X[i][-24:]))
        maxlen.append(len(X[i]))

    tmp = np.array(tmp).T
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2]).T
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    del tmp
    
    x = []
    for i in X:
        mean = np.mean(i, axis = 0)
        mins = np.amin(i, axis = 0)
        maxs = np.amax(i, axis = 0)
        stds = np.std(i, axis = 0)
        skews = skew(i, axis=0)
        samples = len(i)
        vec = np.concatenate([mins, mean, maxs, stds, skews])
        vec = np.append(vec, samples)
        x.append(vec)
    x = np.array(x)
    
    return (x)
    
def run_experiment(x, y, model, task):
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    data = {}
            
    start = time.time(); count = 0
    if (task == 'multiclass') :
        tmp = y[:, -1]

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
        y_train, y_test = y[train_index], y[test_index]
        
        if task == 'binary':
            xs, ys = balanced_subsample(X_train, y_train, 1.0)
            ys = np.array([[i] for i in ys])
        else:
            xs, ys = shuffle(X_train, y_train, random_state =8)
        
        for i in range(100):
            model.fit(xs, ys)
        y_pred = model.predict(xs)
        yhat = model.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        tr_roc_auc = dict()
        f1= dict()
        te_roc_auc = dict()
        te_matrix = dict()
        
        if (task == 'multiclass'):
            for idx in range(y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(ys[:, idx], y_pred[:, idx])
                tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                
            fpr["micro"], tpr["micro"], _ = roc_curve(ys.ravel(), y_pred.ravel())                
            tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   
            
            for idx in range(y[0].shape[0]):
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
            te_matrix = confusion_matrix(y_test, np.array([round(i) for i in yhat]))
            f1 = f1_score(y_test, np.array([round(i) for i in yhat]))
            
            fpr, tpr, _ = roc_curve(y_test, yhat)
            te_roc_auc = auc(fpr, tpr)
        
        data[count]['tr_auc'].append(tr_roc_auc)
        data[count]['f1_score'].append(f1)
        data[count]['te_auc'].append(te_roc_auc)
        data[count]['te_matrix'].append(te_matrix)
    
    return (data, model)
       
def r_search(x, y):
    #random search params
    lr_params = {'penalty': ['l1', 'l2'], 'C': sp_rand(1e-5, .1)}
    rf_params = {'criterion': ['gini', 'entropy'], 'n_estimators': sp_randint(50, 500), 'bootstrap': [True, False] }
    gbc_params = {'learning_rate': sp_rand(1e-6, 1e-1), 'n_estimators': sp_randint(50, 500), 'loss': ['deviance', 'exponential']}
    mlp_params = {'hidden_layer_sizes':[(64, 64), (128, 128), (256, 256), (512, 512)], 'alpha': sp_rand(1e-6, 1e-2)}
    #svm_params = {'kernel': ['rbf', 'linear'], 'C':sp_rand (10, 1e5)} 

    data = {}
    xs, ys = balanced_subsample(x, y)
    lst = [LR(verbose = 1, max_iter = 1000), RF(verbose = 1), MLP(verbose = True), GBC(verbose = 1)]
    names = ['LR', 'RF', 'MLP', 'GB']
    params = [lr_params, rf_params, mlp_params, gbc_params]
    for idx in range(len(lst)):
        n_iter_search = 60
        start = time.time()    
        rsearch = random_search(estimator = lst[idx], param_distributions = params[idx], n_iter=n_iter_search, scoring='roc_auc', fit_params=None, n_jobs=1, iid=True, refit=True, cv=5, verbose=0, random_state=8)
        rsearch.fit(xs, ys)
        data[names[idx]] = rsearch.cv_results_
        print (names[idx]+" results complete.")
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time.time() - start), n_iter_search))
    return (data)
            
    
def balanced_subsample(x,y,subsample_size=1.0):

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
    
def discretize (X, qc):
    sentences = {}
    for h in list(X.keys()):
        sentences[h] = []
        for timestep in X[h]:
            for idx in range(len(timestep)):
                q = qc[idx]
                if timestep[idx] <= q[1]: string = '_1'
                elif q[1] < timestep[idx] <= q[2]: string = '_2'
                elif q[2]<timestep[idx] <=q[3]: string = '_3'
                elif q[3] <timestep[idx]<=q[4]: string = '_4'
                elif timestep[idx] > q[4]: string = '_5'
                else: print(timestep[idx])
                sentences[h].append(str(idx) + string)
    return (sentences)

def onehot(dix, size=942):
    #make dx history one-hot
    vec = []
    for i in dix:
        tmp = [0] * size
        for j in i:
            tmp[j] +=1
        vec.append(tmp)
    return vec

def demographics_make(Demo):
    ages = Demo[:,3]
    ages = [float(a) for a in ages]
    
    hist, bins = np.histogram(ages, bins=4)
    ages=np.digitize(ages, bins)
    ages = [a-1 for a in ages]
    
    marital = {'MARRIED': 0, 'SEPARATED': 1, 'SINGLE':2, 'WIDOWED':3, 'DIVORCED': 4, 'LIFE PARTNER':5, 'UNKNOWN (DEFAULT)': 6, 'nan': 6}
    ethn = {'AMERI': 0, 'ASIAN': 1, 'BLACK': 2, 'CARIB': 3, 'HISPA': 4, 'MIDDL': 5, 'MULTI': 6, 'NATIV': 7, 'OTHER': 12, 'PATIE': 12, 'PORTU': 10, 'SOUTH': 11, 'UNABL': 12, 'UNKNO': 12, 'WHITE': 8}
    insurance = {'Medicaid': 0, 'Government': 1, 'Medicare': 2, 'Self Pay': 3, 'Private': 4}
    
    demo = []

    for idx in range(len(Demo)):
        i = insurance[Demo[idx][0]]
        #los = Demo[idx][1]
        e = ethn[Demo[idx][2]]
        age = ages[idx]
        m = marital[Demo[idx][4]]
        demo.append([age, e, i, m])
    
    enc = OneHotEncoder()
    demo = enc.fit_transform(demo).toarray()
    
    #ages = np.array(ages).reshape(len(ages),1)
    #demo = np.concatenate((ages, demo), axis=1)
        
    return (demo, enc.feature_indices_)
    
def reporting(dat):
    data = {};
    
    #LOS
    task = 'LOS'
    data[task] = {}
    for m in list(dat.keys()):
        for f in list(dat[m].keys()):
            data[task][m+'_'+f] = {}
            f1 = []; auc = []
            sen = {}
            spec = {}
            for loop in list(dat[m][f].keys()):
                sen[loop]= []
                spec[loop] = []
                f1.append(list(dat[m][f][loop]['f1_score'][0].values()))
                auc.append(list(dat[m][f][loop]['te_auc'][0].values()))
                for mat in list(dat[m][f][loop]['te_matrix'][0].values()):
                    tn, fp, fn, tp = mat.ravel()
                    sen[loop].append(1.0* (tp/(tp+fn)))
                    spec[loop].append(1.0* (tn/(tn+fp)))
            f1 = (np.mean(f1, axis = 0), np.std(f1, axis = 0))
            auc = (np.mean(auc, axis =0), np.std(auc, axis = 0))
            sen = (np.mean(list(sen.values()), axis =0), np.std(list(sen.values()), axis = 0))
            spec = (np.mean(list(spec.values()), axis =0), np.std(list(spec.values()), axis = 0))
            names = [' 1-2', ' 2-3', ' 3-5', ' 5-8', ' 8-14', ' 14-21', ' 21-30', ' 30+', ' micro']
            for n in range(len(names)):
                data[task][m+'_'+f]['f1_' + names[n]] = '{0:.3}'.format(f1[0][n]) + ' ({0:.3})'.format(f1[1][n])
                data[task][m+'_'+f]['auc_' + names[n]] = '{0:.3}'.format(auc[0][n]) + ' ({0:.3})'.format(auc[1][n])
                try:
                    data[task][m+'_'+f]['sen_' + names[n]] = '{0:.3}'.format(sen[0][n]) + ' ({0:.3})'.format(sen[1][n])
                    data[task][m+'_'+f]['spec_' + names[n]] = '{0:.3}'.format(spec[0][n])+ ' ({0:.3})'.format(spec[1][n])
                except: pass
    #readm or mortality
    task = 'readm' #or mortality
    data[task] = {}
    for m in list(dat.keys()):
        for f in list(dat[m].keys()):
            data[task][m+'_'+f] = {}
            f1 = []; auc = []
            sen = []
            spec = []
            for loop in list(dat[m][f].keys()):
                f1.append(dat[m][f][loop]['f1_score'][0])
                auc.append(dat[m][f][loop]['te_auc'][0])
                tn, fp, fn, tp = dat[m][f][loop]['te_matrix'][0].ravel()
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
        
    #pheno25
    task = '25 ddx'
    data[task] = {}
    dct = {0: 'Hypertension', 1: 'CHF', 2: 'Atrial fibrillation', 3: 'Coronary atherosclerosis', 4: 'Acute kidney failure', 5: 'DMII w/o complication', 6: 'Hyperlipidemia NEC/NOS', 7: 'Acute respiratory failure', 8: 'Urinary tract infection NOS', 9: 'Esophageal reflux', 10: 'Pure hypercholesterolemia', 11: 'Anemia NOS', 12: 'Pneumonia, organism NOS', 13: 'Hypothyroidism NOS', 14: 'Acute posthemorrhage anemia', 15: 'Acidosis', 16: 'Chronic airway obstruct NEC', 17: 'Severe sepsis', 18: 'Food/vomit pneumonitis', 19: 'Septicemia NOS', 20: 'Chronic kidney dis NOS', 21: 'Hypertensive chronic kidney disease NOS w/ chronic kidney disease stage I-IV', 22: 'Depressive disorder NEC', 23: 'Thrombocytopenia NOS', 24: 'Tobacco use disorder', 25: 'micro'}

    for m in list(dat.keys()):
        for f in list(dat[m].keys()):
            data[task][m+'_'+f] = {}
            f1 ={}; auc = {}; sen = {}; spec = {}
            for idx in list(dat[m][f].keys()):
                f1[idx]= []; auc[idx] = []
                sen[idx] = []; spec[idx] = []
                for loop in list(dat[m][f][idx]):
                    f1[idx].append(dat[m][f][idx][loop]['f1_score'][0])
                    auc[idx].append(dat[m][f][idx][loop]['te_auc'][0])
                    tn, fp, fn, tp = dat[m][f][idx][loop]['te_matrix'][0].ravel()
                    sen[idx].append(1.0* (tp/(tp+fn)))
                    spec[idx].append(1.0* (tn/(tn+fp)))
            
            f1 = (np.mean(list(f1.values()), axis = 1), np.std(list(f1.values()), axis = 1))
            auc = (np.mean(list(auc.values()), axis =1), np.std(list(auc.values()), axis =1))
            sen = (np.mean(list(sen.values()), axis =1), np.std(list(sen.values()), axis =1))
            spec = (np.mean(list(spec.values()), axis =1), np.std(list(spec.values()), axis =1))
            
            for n in range(len(f1[0])):
                data[task][m+'_'+f]['f1_'+dct[n]] = '{0:.3}'.format(f1[0][n]) + ' ({0:.3})'.format(f1[1][n])
                data[task][m+'_'+f]['auc_'+dct[n]] = '{0:.3}'.format(auc[0][n]) + ' ({0:.3})'.format(auc[1][n])
                try:
                    data[task][m+'_'+f]['sen_' + dct[n]] = '{0:.3}'.format(sen[0][n]) + ' ({0:.3})'.format(sen[1][n])
                    data[task][m+'_'+f]['spec_' + dct[n]] = '{0:.3}'.format(spec[0][n])+ ' ({0:.3})'.format(spec[1][n])
                except:pass
    
def str2bool(v):
    return v.lower() in ("True", "true", "yes", "Yes", 't', 'T', '1', 'YES', 'Y', 'y')
    
if __name__ == '__main__':
    xs = []; dxs = []; maxlens = []
    
    parser = argparse.ArgumentParser("Input X, Y, and settings.")
    parser.add_argument('--x', type = str, action = 'append', help = "Enter X's.")
    parser.add_argument('--x_name', type = str, action = 'append', help = "Enter X names.")
    parser.add_argument('--y', type = str, help = 'Enter task w/ labels Y.')
    parser.add_argument('--mode', choices = ['binary', 'multilabel', 'multiclass'])
    parser.add_argument('--o', help = "Enter Output File name.")
    args = parser.parse_args("--x /home/andy/Desktop/MIMIC/vars/npy/seqs/X19.npy --x /home/andy/Desktop/MIMIC/vars/npy/seqs/sentences.npy --x_name 19ts --x_name w2v --x_name sentences --y /home/andy/Desktop/MIMIC/vars/npy/Ys/Yr.npy --w2v False --w2v True --w2v True --maxlen None --maxlen 39 --maxlen 1000 --multi --mode lstm".split())
    #args = parser.parse_args()

    try:
        with open(args.x, 'rb') as f:
            xs = pickle.load(f)
    except:
        x = np.load(args.x)
        xs = [x]       
    
    for m in args.maxlen:
        try: maxlens.append(int(m))
        except: maxlens.append(None)
        
    y = np.load(args.y)    
    
    data = main(xs = xs, x_names = args.x_name, y= y, task = args.mode )
    
    with open(args.o, 'wb') as f:
        pickle.dump(data, f, protocol = 2)

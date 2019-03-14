#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 00:00:52 2018

@author: af1tang
"""
import pickle
import argparse
import pandas as pd
import numpy as np
import progressbar
from datetime import datetime, timedelta

from utilities import *

##### Labels #####
def make_labels():
    icu_details = pd.read_csv(path_views + '/icustay_detail.csv')
    #apply exclusion criterias
    icu_details = icu_details[(icu_details.age>=18)&(icu_details.los_hospital>=1)&(icu_details.los_icu>=1)]
    subj = list(set(icu_details.subject_id.tolist()))
    #make pivot tables for ICD-9 
    print("="*80)
    print("Making pivot table for ICD-9 codes.".center(80))
    print("="*80)
    dx_dct, dx_freq = pivot_icd(subj)
    top25 = dx_freq[0:19] + dx_freq[20:26]
    top25= [i[0] for i in top25]
    icd2idx = dict([(v,k) for k,v in enumerate(top25)])
    #make labels
    dct = {}
    bins = np.array([1, 2, 3, 5, 8, 14, 21, 30, 5000])
    print('Done!')
    print("="*80)
    print("Generating Labels...".center(80))
    print("="*80)
    for sample in progressbar.progressbar(range(len(subj))):
        s = subj[sample]
        lst = icu_details[icu_details.subject_id==s].hadm_id.tolist()
        
        times = [(pd.to_datetime(icu_details[icu_details.hadm_id==i].admittime.values[0]),
                  pd.to_datetime(icu_details[icu_details.hadm_id==i].dischtime.values[0]), i) for i in lst]
        times = list(set(times))
        times = sorted(times, key= lambda x: x[0])
    
        readmit = 0
        for t1,t2 in pairwise(iterable = times):
            difference = (t2[0] - t1[1]).days
            if difference <=30:
                hadm = t1[-1]
                readmit =1
            if difference < 0:
                print (difference, s)
        if readmit == 0:
            morts = [(icu_details[icu_details.hadm_id == h[-1]].hospital_expire_flag.values[0], h[-1]) for h in times]
            hadm = [m[-1] for m in morts if m[0] ==1]
            if len(hadm) >1:
                print (morts)   #error, one can only experience mortlaity once
            elif len(hadm)==1:
                hadm = hadm[0]  #pick the mortality stay if no readmission
            else:
                lengths = [(t[1] - t[0],t[-1]) for t in times]
                hadm = sorted(lengths, key = lambda x: x[0])[-1][-1]    #pick the longest stay if no readmit and no deaths.
        
        #digitize los
        los_bin = np.digitize(icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0], bins)
        #diagnostic labels
        dx_labels = [note for note in dx_dct[s][hadm] if note in top25]
        ohv = np.sum(one_hot([icd2idx[note] for note in dx_labels], 25), axis=0)
        dct[s] = {'hadm_id': hadm, 'readmit': readmit,
                   'los_hospital': icu_details[(icu_details.hadm_id == hadm)].los_hospital.values[0],
                   'los_bin': los_bin,
                   'mort': icu_details[icu_details.hadm_id == hadm].hospital_expire_flag.values[0],
                   'dx_lst': dx_dct[s][hadm],
                   'dx': ohv }
    return dct


##### Diagnosis Pivot Table ######
def pivot_icd(subj):
    '''subj: list of cohort subject_ids'''
    df = pd.read_csv(path_tables + '/diagnoses_icd.csv')
    #icd names
    icd_names = pd.read_csv(path_tables + '/d_icd_diagnoses.csv')
    #make dictionary of icd9 codes
    dct = {}
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        dictionary = df[(df.subject_id == s)][['hadm_id', 'icd9_code']].groupby('hadm_id')['icd9_code'].apply(list).to_dict()
        dictionary = dict([(k,v ) for k,v in dictionary.items()])
        dct[s] = dictionary
    lengths = [dct[i].values() for i in dct.keys()]
    lengths = flatten(lengths)
    lengths = flatten(lengths)
    unique, counts = np.unique(lengths, return_counts=True)
    #frequency dictionary
    dct_freq = dict(zip(unique, counts))
    items = sorted(dct_freq.items(), key = lambda x: x[1], reverse = True)
    ## add names ##
    common = list(set(icd_names.icd9_code).intersection([i[0] for i in items]))
    common = icd_names[icd_names.icd9_code.isin(common)]
    common = common[['icd9_code', 'short_title']].groupby('icd9_code')['short_title'].apply(list).to_dict()
    dct_freq = []
    for idx, count in items:
        if idx in common.keys():
            dct_freq.append((idx, common[idx][0], count))
    return dct, dct_freq

#### Features ####
def get_features(patients):
    '''patients: {subject_id: hadm_id}'''
    p_bg = pd.read_csv(path_views + '/pivoted_bg.csv')
    #p_gcs = pd.read_csv(path_views + '/pivoted_gcs.csv')
    #p_gcs = p_gcs[['icustay_id', 'charttime', 'gcs']]
    #p_uo = pd.read_csv(path_views+'/pivoted_uo.csv')
    p_vital= pd.read_csv(path_views + '/pivoted_vital.csv')
    p_lab = pd.read_csv(path_views + '/pivoted_lab.csv')
    cohort = pd.read_csv(path_views + '/icustay_detail.csv')
    ## Exclusion criteria ##
    cohort = cohort[cohort.subject_id.isin(patients.keys())&(cohort.hadm_id.isin(patients.values()))]

    ## hourly binning ##
    p_bg.charttime = pd.to_datetime(p_bg.charttime)
    p_bg = p_bg.dropna(subset=['hadm_id'])
    p_vital.charttime = pd.to_datetime(p_vital.charttime)
    p_vital = p_vital.dropna(subset=['icustay_id'])
    p_lab.charttime = pd.to_datetime(p_lab.charttime)
    p_lab = p_lab.dropna(subset=['hadm_id'])
    #p_uo.charttime = pd.to_datetime(p_uo.charttime)
    #p_uo = p_uo.dropna(subset=['icustay_id'])
    #p_gcs.charttime = pd.to_datetime(p_gcs.charttime)
    #p_gcs = p_gcs.dropna(subset=['icustay_id'])
    
    ## initialize icustays dict ##
    dct_bins = {}
    lst= sorted(list(set(cohort.hadm_id)))
    hadm_dct = dict([(h, cohort[cohort['hadm_id']==h].subject_id.values[0]) for h in lst])
    
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    ref_ranges = [83.757076260234783, 118.82208983706279, 61.950770137747298, 
                  36.73772, 18.567563025210085, 96.941729323308266, 90.0,
                  4.5, 12.5, 0.89999999999999991, 140.5, 25.0, 275.0, 1.61,
                  4.25, 1.140, 7.4000000000000004, 39.0,1.5]
    dfs = [p_vital, p_lab, p_bg]
    lsts = [icustays, lst, lst]
    cols = [['heartrate', 'sysbp', 'diasbp', 'tempc', 'resprate', 'spo2', 'glucose'],
                   ['albumin', 'bun','creatinine', 'sodium', 'bicarbonate', 'platelet', 'inr'], 
                   ['potassium', 'calcium', 'ph', 'pco2', 'lactate']]
    
    ## initialize features by filtered hadm ##
    features = {}
    subj = sorted(set(cohort.subject_id))

    print("Initializing Timesteps..." )
    print("........")
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        hadm = patients[s]
        timesteps = [pd.to_datetime(datetime.strptime(cohort[cohort.hadm_id==hadm].admittime.values[0], 
                                          '%Y-%m-%d %H:%M:%S') + timedelta(hours=hr)) for hr in range(48)]
        timesteps = [tt.replace(microsecond=0,second=0,minute=0) for tt in timesteps]
        features[hadm] = {}
        for t in timesteps:
            features[hadm][t] = {}

    print()
    print("Eliminating samples with too few timesteps...")
    ## eliminate low time-step samples ##
    lst = []
    #initialize timestamps with vital signs
    for j in progressbar.progressbar(range(len(icustays))):
        h = icustays[j]
        if icu_dct[h] in features.keys():
            timesteps = [i for i in p_vital[p_vital['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[icu_dct[h]].keys())]
            if len(timesteps) >= 6:
                lst.append(icu_dct[h])

    #get timestamps for labs
    lst2 = []
    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_lab[p_lab['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[h].keys())]
        if len(timesteps)>=1:
            lst2.append(h)
    lst = lst2; del lst2
    #update icustays list and features
    features = dict([(k,v) for k,v in features.items() if k in lst])
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())

    print()
    print("="*80)
    print("Generating Timeseries Features")
    print("="*80)
    print()
    lsts = [icustays, lst, lst]
    feature_index=0
    for idx in range(len(dfs)):
        for c in cols[idx]:
            #dfs[idx][c] = (dfs[idx][c]-dfs[idx][c].min() )/ (dfs[idx][c].quantile(.95) - dfs[idx][c].min())
            top5 = dfs[idx][c].quantile(.95) 
            bot5 = dfs[idx][c].quantile(.05) 
            #dfs[idx][c][dfs[idx][c] >=1] = 1
            print('{0}: {1}'.format( c, ref_ranges[feature_index]))
            print()
            #for each admission, for each hourly bin, construct feature vector
            for i in progressbar.progressbar(range(len(lsts[idx]))):
                h = lsts[idx][i]
                if len(lst) == len(lsts[idx]):
                    s = dfs[idx][dfs[idx]['hadm_id']==h].set_index('charttime')[c]
                else:
                    s =  dfs[idx][dfs[idx]['icustay_id']==h].set_index('charttime')[c]
                    h = icu_dct[h]
                
                s = s.interpolate(limit_direction = 'both', limit_area = 'inside')
                s = s.fillna(ref_ranges[feature_index])
                time_range= sorted(features[h].keys())
                s = s.loc[time_range[0]: time_range[-1]]
                if len(s)>0:
                    s= s.resample('H').ohlc()['close'].interpolate(limit_direction='both')

                    s = s.reindex(pd.to_datetime(time_range))
                    s = s.interpolate()
                    s = s.fillna(ref_ranges[feature_index])
                    s[s>=top5] = top5
                    s[s<=bot5] = bot5
                    s = dict([(key,val) for key,val in s.items() if key <= max(features[h].keys())])
                    times = sorted(s.keys())
                    for t in time_range:
                        if t < times[0]:
                            features[h][t][c] = s[times[0]]
                        elif t in times:
                            features[h][t][c] = s[t]
                        elif t not in s.keys():
                            curr = find_nearest(times, t)
                            features[h][t][c] = s[curr]
                            s[t] = s[curr]
                        else:
                            print(times, t)
                    if pd.isnull(list(s.values())).any():
                        print(s)
                else:
                    for t in sorted(features[h].keys()):
                        features[h][t][c] = ref_ranges[feature_index]
            feature_index+=1
    return features

def get_demographics(patients):
    '''patients: {subject_id: hadm_id}
    post: creates demographics dictionary by subject_id, and index dictionary'''
    from sklearn.preprocessing import LabelEncoder
    subj = list(set(patients.keys()))
    hadm = list(set(patients.values()))
    cohort = pd.read_csv(path_views + '/icustay_detail.csv')
    ## Exclusion criteria ##
    cohort = cohort[cohort.subject_id.isin(patients.keys())&(cohort.hadm_id.isin(patients.values()))]
    admissions = pd.read_csv(path_tables + '/admissions.csv')
    cohort = cohort[['subject_id', 'hadm_id', 'age', 'ethnicity']]
    admissions = admissions[['subject_id', 'hadm_id', 'discharge_location', 'marital_status', 'insurance' ]]
    df = pd.merge(cohort, admissions, on = ['subject_id', 'hadm_id'])
    df = df.drop_duplicates()
    df = df[(df.subject_id.isin(subj) & (df.hadm_id.isin(hadm)) )]    
    #discretize and to dict
    df = df.set_index('subject_id')
    df = df.drop(columns = ['hadm_id'])
    df['age'] = pd.qcut(df.age, 5, ['very-young', 'young', 'normal', 'old', 'very-old'])
    df['marital_status'] = df['marital_status'].fillna(value = 'UNKNOWN MARITAL')
    dct = df.to_dict('index')
    dct = dict([(k, list(set(v.values()))) for k,v in dct.items()])
    #label encoding
    categories = list(set(flatten([list(df[c].unique()) for c in list(df.columns)]) ))
    encoder = LabelEncoder()
    encoder.fit(categories)
    #label encode the dictionary
    dct = dict([(k, encoder.transform(v) ) for k,v in dct.items()])
    category_dict = dict([(encoder.transform([c])[0], c) for c in categories])
    return dct, category_dict

def auxiliary_features(y, Z, sentences, words):
    import gensim
    vocab = list(set(flatten(sentences)))
    vocab = list(set([w[0:3] for w in vocab]))
    vocab = dict([(v,k) for k,v in enumerate(vocab)])
    onehot = [np.sum(one_hot(list(set([vocab[w[0:3]] for w in sentences[i] if w[0:3] in vocab.keys()] )), len(vocab) ), axis=0)  for i in range(len(y))]
    full_sentences = [sentences[i] + words[i] for i in range(len(y))]
    dx_demo = [sentences[i] + [str(zz) for zz in Z[i]] for i in range(len(y))]
    print("Wait for the skip gram .... (w2v)")
    w2v, _,_,_ = skip_gram(sentences)
    print("and more skip gram .... (h2v)")
    h2v, _,_,_ = skip_gram(dx_demo)
    print("and even more skip gram .... (sentences)")
    sent_vecs, _, _,_ = skip_gram(full_sentences)
    #demographics features
    demo_size = max(flatten(Z))+1
    demo = [np.sum(one_hot(zz, demo_size),axis=0 )for zz in Z]
    return np.array(onehot), np.array(w2v), np.array(h2v), np.array(sent_vecs), np.array(demo)

#### Preprocessing ####
def preprocess(features, labels, demographics):
    '''pre: features and labels
    post: X = [[x1, ... xT]_1, ...], y= [(mort, readm, los, dx)] '''
    from sklearn.preprocessing import MinMaxScaler
    subj = list(set(labels.keys()))   
    hadm = list(set(features.keys()))
    col_dict = dict ([(v,k) for k,v in enumerate(features[hadm[0]][list(features[hadm[0]].keys())[0]].keys())])
    cols = sorted(col_dict.keys())
    items = []
    for i in progressbar.progressbar(range(len( subj ) ) ):
        s = subj[i]
        h = labels[s]['hadm_id']
        if h in hadm:
            x = np.zeros((len(features[h].keys()), len(col_dict)))
            for index in range(len(sorted(features[h].keys()))):
                t = sorted(features[h].keys())[index]
                x[index, [col_dict[k] for k in cols]] = [features[h][t][k] for k in cols]
            mort = labels[s]['mort']
            los = list(one_hot([labels[s]['los_bin']], 9)[0])
            readmit = labels[s]['readmit']
            dx = labels[s]['dx']
            y = (mort, readmit, los, dx)
            z = demographics[s]
            #auxiliary features
            x48 = np.concatenate((np.min(x, axis=0), np.max(x, axis=0), np.mean(x,axis=0), np.std(x,axis=0)),axis=-1)            
            sentence = labels[s]['dx_lst']
            items.append((x, y, z, x48, sentence))
    X, y, Z, X48, sentences = zip(*items)
    X, y, Z, X48, sentences = np.array(list(X)), list(y), np.array(list(Z)), np.array(list(X48)), list(sentences)
    #normalize each feature to [0,1]
    words = [[] for i in range(len(X))]
    for i in range(len(X[0,0,:])):
        #add to visit words
        mean, std, minimum, maximum = np.mean(X[:,:,i]), np.std(X[:,:,i]), np.min(X[:,:,i],axis=1), np.max(X[:,:,i], axis=1) 
        arr_min, arr_max = minimum < (mean - std), maximum > (mean + std)
        for j in range(len(arr_min)):
            if arr_min[j]: words[j].append(str(i) + '_low')
            if arr_max[j]: words[j].append(str(i) + '_high')
        #scale X
        scaler = MinMaxScaler()
        x_row = scaler.fit_transform(X[:,:,i])
        X[:,:,i] = x_row
    #transform X48
    scaler = MinMaxScaler()
    X48 = scaler.fit_transform(X48)
    return X, y, Z, X48, sentences, words


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--path_tables', type=str, default='/local_mimic/tables',
                        help='Path to the original MIMIC-III Tables.')
    parser.add_argument('--path_views', type=str, default='/local_mimic/views',
                        help='Path to View tables from MIMIC-III cookbook.')
    parser.add_argument('--path_save', type=str, default='/local_mimic/save',
                        help='Set the directory to store features, labels and such.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    
    path_tables = opts.path_tables
    path_views = opts.path_views
    path_save = opts.path_save

    if (path_tables and path_views and path_save):
        create_dir_if_not_exists(path_save)
    
        labels = make_labels()
        with open(path_save + '/labels', 'wb') as f:
            pickle.dump(labels, f)
        print("Saving labels ..." )
        print("........")
        print("Done!")
        print("Constructing Feature Space ..... ")
        print("........")
        patients = dict([(k, labels[k]['hadm_id']) for k in labels.keys()])
        features = get_features(patients)
        patients = dict([(k,v) for k,v in patients.items() if v in features.keys()])
        demographics, categories = get_demographics(patients)
        with open(path_save + '/features', 'wb') as f:
            pickle.dump(features,f)
        with open(path_save + '/demographics', 'wb') as f:
            pickle.dump(demographics, f)
        with open(path_save + '/categories', 'wb') as f:
            pickle.dump(categories, f)
        print("Done!")
        print()
        print("Preprocessing to construct X and y.")
        X, y, Z, X48, sentences, words = preprocess(features, labels, demographics)
        np.save(path_save + '/X19', X)
        np.save(path_save+'/Z', Z)
        np.save(path_save + '/X48', X48)
        with open(path_save+'/y', 'wb') as f:
            pickle.dump(y, f)
        with open(path_save+'/history_level', 'wb') as f:
            pickle.dump(sentences, f)
        with open(path_save+'/visit_level', 'wb') as f:
            pickle.dump(words, f)   
        print("Making auxiliary features: onehot, w2v, h2v, sentences ... etc.")
        onehot, w2v, h2v, sent_vecs, demo = auxiliary_features(y, Z, sentences, words)
        np.save(path_save+'/onehot', onehot)
        np.save(path_save+'/w2v', w2v)
        np.save(path_save+'/h2v', h2v)
        np.save(path_save+'/sentences', sent_vecs)
        np.save(path_save+'/demo', demo)
        print("Done!")
    else:
        print("Make sure you have the MIMIC-III Tables and Views.")
        print("See Requirements page.")
        

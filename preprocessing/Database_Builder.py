# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:07:54 2016

@author: af1tang
"""
import sys, pickle
import os.path as path

#import sqlite3
import csv
import gzip
import MySQLdb as mysql
#import jaydebeapi as jdbc
import pandas as pd
#from pandas import DataFrame
from pandas.io import sql as transfer


#import numpy as np
#import math
import datetime
import re
#import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from tempfile import mkdtemp


    
#def main():
    
#    print ("++++++++++++++++++++++++")
    
#    print ("Creating tables under MySQL database.")
#    make_sql()
#    print ("A New UFM Table will be made on the MySQL server.")
#    make_ufm()    
#    make_demo()
'''Size of df: 2288146
Total rows: 161254506
'''

#####################################################################
##### Part 0. Creating the MySQL Database for the greater good. #####
#####################################################################

def make_sql(conn, admissions_doc, diagnoses_doc, icds_doc, procedures_doc, labevents_doc, items_doc, labitems_doc, patients_doc):
    #MySQLdb
    #conn = mysql.connect (host=host, user=user, passwd=pw, db=mimic, port=port)
    c = conn.cursor()

    #JDBC connect
    #conn = jdbc.connect('com.mysql.jdbc.Driver', ['jdbc:mysql//illidan-gpu-1.egr.msu.edu:3306', 'af1tang', 'illidan'])
    #c = conn.cursor()
    
    ## create tables ##
    c.execute('DROP TABLE IF EXISTS admissions')
    c.execute('DROP TABLE IF EXISTS diagnoses')
    c.execute('DROP TABLE IF EXISTS icds')
    c.execute('DROP TABLE IF EXISTS labevents')
    c.execute('DROP TABLE IF EXISTS procedureevents')
    c.execute('DROP TABLE IF EXISTS patients')
    c.execute('DROP TABLE IF EXISTS items')
    c.execute('DROP TABLE IF EXISTS labitems')

    
    c.execute('CREATE TABLE IF NOT EXISTS admissions(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ADMITTIME DATETIME, DISCHTIME DATETIME, DEATHTIME DATETIME, ADMISSION_TYPE TEXT, ADMISSION_LOCATION TEXT, DISCHARGE_LOCATION TEXT, INSURANCE TEXT, LANGUAGE TEXT, RELIGION TEXT, MARITAL_STATUS TEXT, ETHNICITY TEXT, EDREGTIME DATETIME, EDOUTTIME DATETIME, DIAGNOSIS TEXT, HOSPITAL_EXPIRE_FLAG INT, HAS_IOEVENTS_DATA INT, HAS_CHARTEVENTS_DATA INT);')    
    #c.execute('CREATE TABLE IF NOT EXISTS chartevents')
    #c.execute('CREATE TABLE IF NOT EXISTS labitems')
    c.execute('CREATE TABLE IF NOT EXISTS diagnoses(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, SEQ_NUM INT, ICD9_CODE TEXT);')
    c.execute('CREATE TABLE IF NOT EXISTS icds(ROW_ID INT, ICD9_CODE TEXT, SHORT_TITLE TEXT, LONG_TITLE TEXT);')
    #c.execute('CREATE TABLE IF NOT EXISTS inputevents')
    #c.execute('CREATE TABLE IF NOT EXISTS outputevents')
    #c.execute('CREATE TABLE IF NOT EXISTS procedures')
    c.execute('CREATE TABLE IF NOT EXISTS labevents(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ITEMID INT, CHARTTIME DATETIME, VALUE TEXT, VALUENUM REAL, VALUEUOM TEXT, FLAG TEXT);')
    #c.execute('CREATE TABLE IF NOT EXISTS items')
    c.execute('CREATE TABLE IF NOT EXISTS procedureevents(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ICUSTAY_ID INT, STARTTIME DATETIME, ENDTIME DATETIME, ITEMID INT, VALUE REAL, VALUEUOM TEXT, LOCATION TEXT, LOCATIONCATEGORY TEXT, STORETIME DATETIME, CGID INT, ORDERID INT, LINKORDERID INT, ORDERCATEGORYNAME TEXT, SECONDARYORDERCATEGORYNAME TEXT, ORDERCATEGORYDESCRIPTION TEXT, ISOPENBAG INT, CONTINUEINNEXTDEPT INT, CANCELREASON INT, STATUSDESCRIPTION TEXT, COMMENTS_EDITEDBY TEXT, COMMENTS_CANCELEDBY TEXT, COMMENTS_DATE DATETIME);')
    c.execute('CREATE TABLE IF NOT EXISTS items(ROW_ID INT, ITEMID INT, LABEL TEXT, ABBREVIATION TEXT, DBSOURCE TEXT, LINKSTO TEXT, CATEGORY TEXT, UNITNAME TEXT, PARAM_TYPE TEXT, CONCEPTID INT);')
    c.execute('CREATE TABLE IF NOT EXISTS labitems(ROW_ID INT, ITEMID INT, LABEL TEXT, FLUID TEXT, CATEGORY TEXT, LOINC_CODE TEXT);')
    c.execute('CREATE TABLE IF NOT EXISTS patients(ROW_ID INT, SUBJECT_ID INT, GENDER TEXT, DOB DATETIME, DOD DATETIME, DOD_HOSP DATETIME, DOD_SSN DATETIME, EXPIRE_FLAG TEXT);')    
    
        ##import admissions table
    with gzip.open(admissions_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        #to_db = [(i['c1'], i['c2'], i['c3'], i['c4'], i['c5'], i['c6'], i['c7'], i['c8'], i['c9'], i['c10'], i['c11'], i['c12'], i['c13'], i['c14'], i['c15'], i['c16'], i['c17'], i['c18'], i['c19'], i['c20']) for i in dr]
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ADMITTIME'], i['DISCHTIME'], i['DEATHTIME'], i['ADMISSION_TYPE'], i['ADMISSION_LOCATION'], i['DISCHARGE_LOCATION'], i['INSURANCE'], i['LANGUAGE'], i['RELIGION'], i['MARITAL_STATUS'], i['ETHNICITY'], i['EDREGTIME'], i['EDOUTTIME'], i['DIAGNOSIS'], i['HOSPITAL_EXPIRE_FLAG'], i['HAS_IOEVENTS_DATA'], i['HAS_CHARTEVENTS_DATA']) for i in dr]
    try:
        c.executemany("INSERT INTO admissions(ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_IOEVENTS_DATA, HAS_CHARTEVENTS_DATA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Admissions Table.")
    
    #import diagnoses table
    with gzip.open(diagnoses_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        #to_db = [(i['c1'], i['c2'], i['c3'], i['c4'], i['c5'], i['c6'], i['c7'], i['c8'], i['c9'], i['c10'], i['c11'], i['c12'], i['c13'], i['c14'], i['c15'], i['c16'], i['c17'], i['c18'], i['c19'], i['c20']) for i in dr]
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['SEQ_NUM'], i['ICD9_CODE']) for i in dr]
        lst = [i[3] for i in to_db]
        lst = [None for i in lst if i == '']
    try:
        c.executemany("INSERT INTO diagnoses(ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE) VALUES (%s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Diagnoses Table.")
    
    #import icds
    with gzip.open(icds_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['ICD9_CODE'], i['SHORT_TITLE'], i['LONG_TITLE']) for i in dr]
    try:
        c.executemany("INSERT INTO icds(ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE) VALUES (%s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at ICD9 Reference Table.")
        
    #import labevents table
    in_csv = labevents_doc
    chunksize = 100000
    with gzip.open(in_csv, 'r') as f:
        for numlines,l in enumerate (f): pass
    numlines +=1
    for i in range (0, numlines, chunksize) :
        dr = pd.read_csv(in_csv, header=None, nrows = chunksize, skiprows = i)
        columns = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG']
        dr.columns = columns
       # dtypes = {'ROW_ID': int, 'SUBJECT_ID': int, 'HADM_ID': int, 'ITEMID': int, 'CHARTTIME': time, 'VALUE': str, 'VALUENUM': float,  'FLAG': str}
        transfer.to_sql(dr, name = 'labevents', con=conn, index=False, index_label = 'ROW_ID', if_exists = 'append', flavor = 'mysql')        
        print (i)
        #to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ITEMID'], i['CHARTTIME'], i['VALUE'], i['VALUENUM'], i['VALUEUOM'], i['FLAG']) for i in dr]
    #c.executemany("INSERT INTO labevents(ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import procedure events table
    with gzip.open(procedures_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ICUSTAY_ID'], i['STARTTIME'], i['ENDTIME'], i['ITEMID'], i['VALUE'], i['VALUEUOM'], i['LOCATION'], i['LOCATIONCATEGORY'], i['STORETIME'], i['CGID'], i['ORDERID'], i['LINKORDERID'], i['ORDERCATEGORYNAME'], i['SECONDARYORDERCATEGORYNAME'], i['ORDERCATEGORYDESCRIPTION'], i['ISOPENBAG'], i['CONTINUEINNEXTDEPT'], i['CANCELREASON'], i['STATUSDESCRIPTION'], i['COMMENTS_EDITEDBY'], i['COMMENTS_CANCELEDBY'], i['COMMENTS_DATE']) for i in dr]
    try:
        c.executemany("INSERT INTO procedureevents(ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTTIME, ENDTIME, ITEMID, VALUE, VALUEUOM, LOCATION, LOCATIONCATEGORY, STORETIME, CGID, ORDERID, LINKORDERID, ORDERCATEGORYNAME, SECONDARYORDERCATEGORYNAME, ORDERCATEGORYDESCRIPTION, ISOPENBAG, CONTINUEINNEXTDEPT, CANCELREASON, STATUSDESCRIPTION, COMMENTS_EDITEDBY, COMMENTS_CANCELEDBY, COMMENTS_DATE) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Procedure Table.")
        
    #import Items table
    with gzip.open(items_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['ITEMID'], i['LABEL'], i['ABBREVIATION'], i['DBSOURCE'], i['LINKSTO'], i['CATEGORY'], i['UNITNAME'], i['PARAM_TYPE'], i['CONCEPTID']) for i in dr]
    try:
        c.executemany("INSERT INTO items(ROW_ID, ITEMID, LABEL, ABBREVIATION, DBSOURCE, LINKSTO, CATEGORY, UNITNAME, PARAM_TYPE, CONCEPTID) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Items table.")
    
    ##import labitems table
    with gzip.open(labitems_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['ITEMID'], i['LABEL'], i['FLUID'], i['CATEGORY'], i['LOINC_CODE']) for i in dr]
    try:
        c.executemany("INSERT INTO labitems(ROW_ID, ITEMID, LABEL, FLUID, CATEGORY, LOINC_CODE) VALUES (%s, %s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Lab Items Reference.")
    
    ##import patients table
    with gzip.open(patients_doc,'r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['GENDER'], i['DOB'], i['DOD'], i['DOD_HOSP'], i['DOD_SSN'], i['EXPIRE_FLAG']) for i in dr]
    try:
        c.executemany("INSERT INTO patients(ROW_ID, SUBJECT_ID, GENDER, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);", to_db)
        conn.commit()
    except:
        conn.rollback()
        print ("Error at Patients table.")
    
    print ("Complete.")

####################################################################
##### Part 1. INITIALIZEE UFMs. Skip if already exists. ############
####################################################################
  
def make_ufm (conn, engine):
   
    #connect to sql
    #conn = sqlite3.connect(mimic_doc)
    #c = conn.cursor()
   
    #connect to mysql
    #conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)
    c = conn.cursor()
   
    #JDBC connect
    #conn = jdbc.connect('com.mysql.jdbc.Driver', ['jdbc:mysql//illidan-gpu-1.egr.msu.edu:3306', 'af1tang', 'illidan'])
    #c = conn.cursor()
    
    c.execute('DROP TABLE IF EXISTS UFM')

    #make flags
    #sql_flags = "SELECT DISTINCT ITEMID, FLAG FROM labevents WHERE FLAG is not null or FLAG=''"
    #flags = pd.read_sql_query(sql=sql_flags, con = conn)
    #flags = list(flags['ITEMID'])
    #flags = list(set(flags[1:]))
    #print ("Flags made.")
    
    df = pd.read_sql(sql = "SELECT * FROM admissions", con=conn)
    subjects = dict(Counter(df["SUBJECT_ID"])) #creates Counter for each unique subject
    subj = list(subjects.keys())
    subj = [str(i) for i in subj]
    #flags = [str(i) for i in flags]    
    
############# OPTION 1 ################
    cut = int(len(subj)/75)
    print ('\n'+"+++++++++ IN PROGRESS +++++++++")
    
    #make HDFStore of UFM Table
    #store = pd.HDFStore('UFM_table.h5')
    rows = 0    
    
    for i in range (0,75):
        s = subj[i*cut:((i+1)*cut)]
        print ("Cycle number: {0}, Offset: {1}, Chunk: {2}".format(i, i*cut, (i+1)*cut))
        if i == 74:
            s = subj[i*cut:]
            print ("Ignore above, actual chunksize is {0} to end.".format(i*cut))

        #sql_lab = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE, FLAG FROM labevents WHERE SUBJECT_ID IN ({0}) AND ITEMID IN ({1})".format(','.join('?'*len(s)),','.join('?'*len(flags)))
        sql_lab = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE, FLAG FROM labevents WHERE SUBJECT_ID IN (%s) AND HADM_ID is not null AND HADM_ID !=''" % ','.join(['%s']*len(s))
        sql_proc = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID IN (%s) AND HADM_ID is not null AND HADM_ID !=''" % ','.join(['%s']*len(s))
        sql_dx = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID IN (%s) AND HADM_ID is not null AND HADM_ID !=''"% ','.join(['%s']*len(s))
        sql_dx2 = "SELECT HADM_ID, ADMITTIME AS 'TIME' FROM admissions WHERE SUBJECT_ID IN (%s) AND HADM_ID is not null AND HADM_ID !=''" % ','.join(['%s']*len(s))
        
        #lab_params = tuple(s+flags)
        lab_params = tuple(s)
        other_params = tuple(s)
       
        df_lab = pd.read_sql_query(sql=sql_lab, con = conn, params = lab_params)
        df_proc = pd.read_sql_query(sql=sql_proc, con = conn, params = other_params)    
        df_dx1 = pd.read_sql_query(sql=sql_dx, con = conn, params = other_params)
        df_dx2 = pd.read_sql_query(sql=sql_dx2, con=conn, params = other_params)
        df_dx= pd.merge(df_dx1, df_dx2, how = 'outer', on = 'HADM_ID')
        
        df_dx['VALUE'] = 1
        df_dx['FLAG'] = None
        df_proc['FLAG'] = None
        df_lab['TYPE'] = 'l'
        df_proc['TYPE'] = 'p'
        df_dx['TYPE'] = 'd'
        
        frames = [df_lab, df_proc, df_dx]
        df = pd.concat(frames)
        #store labs, procs and dxs for each patient into HDFStore table
        #for i in s:
        #    temp = df[df['SUBJECT_ID']==int(i)]
        #    store[i] = temp
 
 #Make the dataframe into SQL table:
 
        df.to_sql(name = 'UFM', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'append')
           
        rows += df.size
        print ("Size of df: {0}".format(df.size))
        print ("Total rows: {0}".format(rows))

    c.close()
    conn.close()
    
    print ("UFM TABLES COMPLETE!")
    #return(store)
    
############################################
###### Part 2. Make Demographics Table #####
############################################
    
def make_demo(conn, engine):
#def make_demo(pts):
    #connect to mysql
    #conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS demographics')
    
    #get all patients with specified from patients table
    #sql_patients = "SELECT SUBJECT_ID FROM diagnoses WHERE ICD9_CODE IN (%s)"% ','.join(['%s']*len(pts))
    #patients = pd.read_sql(sql=sql_patients, con = conn, params = tuple(pts))
    #subjects = dict(Counter(patients["SUBJECT_ID"]))
    #subj = list(subjects.keys())
    #subj = [str(i) for i in subj]
    
    #sql = "SELECT SUBJECT_ID, GENDER, DOB, EXPIRE_FLAG FROM patients WHERE SUBJECT_ID IN (%s)".join(['%s']*len(subj))
    sql = "SELECT SUBJECT_ID, GENDER, DOB, EXPIRE_FLAG FROM patients"    
    #df = pd.read_sql(sql=sql, con = conn, params = tuple(subj))
    df = pd.read_sql(sql=sql, con = conn)
    
    #adjust DOB to reflect AGE
    #sql2 = "SELECT SUBJECT_ID, ADMITTIME, ETHNICITY, MARITAL_STATUS, RELIGION from admissions where SUBJECT_ID in (%s)".join(['%s']*len(subj))
    sql2 = "SELECT SUBJECT_ID, ADMITTIME, ETHNICITY, MARITAL_STATUS, RELIGION from admissions"    
    #df2 = pd.read_sql(sql=sql2, con=conn, params= tuple(subj))
    df2 = pd.read_sql(sql=sql2, con=conn)
    lst = list(pd.unique(df2['SUBJECT_ID']))
    
    df['AGE']=0
    df['ETHNICITY'] = ''
    df['MARITAL_STATUS'] = ''
    
    for i in lst:
        marital = list(df2[df2['SUBJECT_ID']==i]['MARITAL_STATUS'])
        eth = list(df2[df2['SUBJECT_ID']==i]['ETHNICITY'])
        dob = list(df[df['SUBJECT_ID']==i]['DOB'])
        first = sorted(list(df2[df2['SUBJECT_ID']==i]['ADMITTIME']))
        #If times are string instead of timestamps, convert to time stamps.
        #t1 = datetime.datetime.strptime(first[0], '%Y-%m-%d %H:%M:%S')
        #t2 = datetime.datetime.strptime(dob[0], '%Y-%m-%d %H:%M:%S')
        t1 = first[0]
        t2 = dob[0]
        age = (t1-t2).days/365.2425
        if age >=300: age = 89
        df.set_value(list(df.SUBJECT_ID[df.SUBJECT_ID==i].index)[0], 'AGE', age)
        df.set_value(list(df.SUBJECT_ID[df.SUBJECT_ID==i].index)[0], 'ETHNICITY', eth[0])
        df.set_value(list(df.SUBJECT_ID[df.SUBJECT_ID==i].index)[0], 'MARITAL_STATUS', marital[0])
    
    df.to_sql(name = 'demographics', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'replace')
    print ("Demographics Table Complete.")  
    
###### Part 3. Fix the Labs ######
##################################
    
def filter_labs(conn):
    #conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)

    sql = "SELECT ITEMID, FLAG, VALUE, VALUEUOM from labevents where FLAG IS NOT NULL AND FLAG != ''"
    flags = pd.read_sql(sql=sql, con = conn)
    keys = list(set(flags.ITEMID))
    
    tab = pd.read_table('/home/andy/Desktop/MIMIC/lab_ref.txt')
    sql = "select * from labitems"
    labs = pd.read_sql(sql= sql, con = conn)
    labs = labs.drop_duplicates()
    
    t = list(set(tab.CLINICAL_CHEMISTRY))
    l = list(set(labs.LABEL))
    r = re.compile(r'\b(?:%s)\b' % '|'.join(t))
    
    intersect = list(filter(r.search,l))
    temp = labs[labs.LABEL.isin(intersect)]
    chosen1s = labs[labs.LABEL.isin(intersect)]
    intersect = temp[temp.ITEMID.isin(keys)]
    
    #return (chosen1s, intersect)
    return (keys)
    
#if __name__ == '__main__':
#    from optparse import OptionParser, OptionGroup
#    desc = "Welcome to UFM Table Maker by af1tang."
#    version = "version 1.0"
#    opt = OptionParser (description = desc, version=version)
#    opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
#    opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
#    (cli, args) = opt.parse_args()
#    opt.print_help()
    
#    mimic = 'MIMIC3'
#    host = 'illidan-gpu-1.egr.msu.edu'
#    user = 'af1tang'
#    pw = 'illidan'    
#    port = 3306
    
#    main()  

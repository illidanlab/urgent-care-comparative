# Predictive Modeling in Urgent Care

This is the code repository for manuscript *"Predictive Modeling in Urgent Care: A Comparative Study of Machine Learning Approaches"* by *Fengyi Tang, Cao Xiao, Fei Wang, Jiayu Zhou*.

## Manuscript Abstract

**Objective**: The growing availability of rich clinical data such as patients' electronic health records (EHR) provide great opportunities to address a broad range of real-world questions in medicine. At the same time, artificial intelligence and machine learning based approaches have shown great premise on extracting insights from those data and helping with various clinical problems. The goal of this study is to conduct a systematic comparative study of different machine learning algorithms for several predictive modeling problems in urgent care. 

**Design**: We assess the performance of four benchmark prediction tasks (e.g., mortality and prediction, differential diagnostics and disease marker discovery) using medical histories, physiological time-series and demographics data from the Medical Information Mart for Intensive Care (MIMIC-III) database.

**Measurements**: For each given task, performance was estimated using standard measures including the area under the receiver operating characteristic (AUC) curve, F-1 score, sensitivity and specificity. Micro-averaged AUC was used for multi-class classification models.

**Results and Discussion**: Our results suggest that recurrent neural networks show the most promise in mortality prediction where temporal patterns in physiologic features alone can capture in-hospital mortality risk (AUC > 0.90). Temporal models did not provide additional benefit compared to deep models in differential diagnostics. When comparing the training-testing behaviors of readmission and mortality models, we illustrate that readmission risk may be independent of patient stability at discharge. We also introduce a multi-class prediction scheme for length of stay which preserves sensitivity and AUC with outliers of increasing duration despite decrease in sample size.

## Usages
### Requirements
* Python 3.4+
* Keras 2.0
* Scikit-Learn
* Gensim
* NumPy
* Pandas
* Tensorflow 1.11+
* Progressbar2
* Postgres (or equivalent for building local MIMIC-III)

### MIMIC-III ###
Please apply for access to the publicly available MIMIC-III DataBase via `https://www.physionet.org/`. 

### Instructions for Use ###

**Workflow**: MIMIC-III Access -> Obtain Views and Tables -> Preprocessing -> Pipeline

1. Obtain access to MIMIC-III and clone this repo to local folder. 
Create a local MIMIC-III folder to store a few files:
* `.../local_mimic`
* `.../local_mimic/views`
* `.../local_mimic/tables`
* `.../local_mimic/save`

These paths will be important for storing views and pivot tables, which will be used for preprocessing.

2. Build MIMIC-III database using `postgres`, follow the instructions outlined in the MIMIC-III repository: 
`https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres`.

3. Go to the pivot folder in the MIMIC-III repository:
`https://github.com/MIT-LCP/mimic-code/tree/master/concepts/pivot`.
Run use the `.sql` scripts to build a local set of `.csv` files of the pivot tables:
* pivoted-bg.sql 
* pivoted_vital.sql
* pivoted_lab.sql
* pivoted_gcs.sql (optional)
* pivoted_uo.sql (optional)

When running the `.sql` script, change the _delimiter_ of the materialized views to ',' when saving as `.csv` file.
Example: 
`mimic=> \copy (select * FROM mimiciii.icustay_detail) to 'icustay_detail.csv' delimiter ',' csv header;`

After running these scripts, you should have obtained local `.csv` files of the pivot tables. 
Create a local folder to place them in, i.e. `.../local_mimic/views/pivoted-bg.csv`. 
Remember this `.../local_mimic/views` folder, as it will be the `path_views` input for preprocessing purposes.

4. Go to the demographics folder in the MIMIC-III repository:
`https://github.com/MIT-LCP/mimic-code/tree/master/concepts/demographics`.

Run `icustay-detail.sql` and obtain a local `.csv` file of `icustays-detail` view. 
Create a local folder to place the `.csv` file in, i.e.`.../local_mimic/views/icustay_details.csv`. 
Again, have this `.csv` file inside the local `views` folder.

A minor change needs to be made in `icustay_details.csv`:  
change `'admission_age' -> 'age'` for the column header in the `.csv` file manually. 

5. Obtain a local copy of the following tables from MIMIC-III:
* admissions.csv
* diagnoses_icd.csv
* d_icd_diagnoses.csv

These can be directly obtained from *Physionet* as compressed files. 
While tables such as `chartevents` are large, the above tables are quite small and easy to query directly if a local copy is available. 

Save these tables under `.../local_mimic/tables` folder. 
Make the following changes: 
* In `~/local_mimic/tables/diagnoses_icd.csv`, change the column titles `"ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"` to
`"row_id","subject_id","hadm_id","seq_num","icd9_code"` (i.e., make lower case). 
* In `~local_mimic/tables/d_icd_diagnoses.csv` change the column titles `"ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"` to
`"row_id","icd9_code","short_title","long_title"` (i.e., again, make lower case). 

6. Run `preprocessing.py` with inputs: 
* `--path_tables <path_tables>`
* `--path_views <path_views>`
* `--path_save <path_save>`.

`<path_tables>` and `<path_views>`  should correspond to the folders under which the local tables and views (pivots and icustays-details) are saved.
 `<path_save>` corresponds to the desired folder to save your variables for training and beyond.
 
 `preprocessing.py` will generate the following files:
 * `X19.npy`: main feature tensor, consisting of time-series data generated from a combination of 19 lab values and vital signs over 48 hour period from start of admissions. 
 * `X48.npy`:  summary feature matrix of the time-series data, with *min*, *mean*, *max*, and *standard-deviation (std)* of each feature as extended features instead of time-series. 
 * `y`: main label matrix, with (mortality_flag, readmission_status, LOS_bin, diagnoses_labels) for each patient. The labels are coupled here, but during `main.py`, user can define which task to pick (i.e. which column of `y`).   
 * `onehot`: one-hot vector of diagnostic history of each patient. This is different than the *top 25 differential diagnosis* task, which is the last column of `y`. Diagnostic history uses *ICD-9 Group Codes* instead of ICD-9 codes (i.e. more general). Used only for mortality, LOS and readmission predictions. 
 * `w2v`: Skip-Gram embeddings for diagnosis histories (auxiliary input). 
 * `h2v`: Skip-Gram embeddings of both diagnostic histories and demographics info (auxiliary input).
 * `demo`: one-hot vector representation of demographics info (auxiliary input).
 * `sentences`: Skip-Gram embeddings of mixed diagnostic histories and abnormal laboratory flags (main feature input).
 
 7. Run `main.py` with selection of features, auxiliary features, task, model, and training conditions:
 * `--features_dir`: path to saved the feature file to use as X. Selections include `X19`, `X48`, `sentences`, or `onehot`.
 * `--auxiliary_dir`: path to auxiliary features to be used for certain models. Selections include `w2v`, `h2v`, or `demo`.
 * `--y_dir`: path to `y`.
 * `--model`: type of model to use for train / test. User can choose among `['lstm', 'cnn', 'mlp', 'svm', 'rf', 'lr','gbc' ]`. LSTM, CNN-LSTM and MLP are deep models, while SVM, random forest (rf), logistic regression (lr) and gradient boost (GBC) are classical models. Note that LSTM and CNN-LSTM need to use `X19` as input feature because they are *temporal models*.  Non-temporal models such as MLP, SVM, rf, lr and gbc should not use `X19`. 
 * `--task`: specifies the learning task. User can choose between `['readmit', 'mort', 'los', 'dx']`.
 * `-checkpoint_dir`: specifies the path to save best models and testing results. 
* `--hidden_size`: specifies number of hidden units for deep models (default =256). 
 * `--learning_rate`: specifies the initial learning rate (default=0.005).
 * `--nepochs`: number of training epochs (default = 100).
 *`--batch_size`: batch size during training (default = 32).

If you find any errors or issues, please do not hesitate to report. 
 

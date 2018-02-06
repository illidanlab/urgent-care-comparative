# Predictive Modeling in Urgent Care

This is the code repository for manuscript `Predictive Modeling in Urgent Care: A Comparative Study of Machine Learning Approaches` by *Fengyi Tang, Cao Xiao, Fei Wang, Jiayu Zhou*. Currently under revision. 

## Manuscript Abstract

**Objective**: The growing availability of rich clinical data such as patients' electronic health records (EHR) provide great opportunities to address a broad range of real-world questions in medicine. At the same time, artificial intelligence and machine learning based approaches have shown great premise on extracting insights from those data and helping with various clinical problems. The goal of this study is to conduct a systematic comparative study of different machine learning algorithms for several predictive modeling problems in urgent care. 

**Design**: We assess the performance of four benchmark prediction tasks (e.g., mortality and prediction, differential diagnostics and disease marker discovery) using medical histories, physiological time-series and demographics data from the Medical Information Mart for Intensive Care (MIMIC-III) database.

**Measurements**: For each given task, performance was estimated using standard measures including the area under the receiver operating characteristic (AUC) curve, F-1 score, sensitivity and specificity. Micro-averaged AUC was used for multi-class classification models.

**Results and Discussion**: Our results suggest that recurrent neural networks show the most promise in mortality prediction where temporal patterns in physiologic features alone can capture in-hospital mortality risk (AUC > 0.90). Temporal models did not provide additional benefit compared to deep models in differential diagnostics. When comparing the training-testing behaviors of readmission and mortality models, we illustrate that readmission risk may be independent of patient stability at discharge. We also introduce a multi-class prediction scheme for length of stay which preserves sensitivity and AUC with outliers of increasing duration despite decrease in sample size.

## Usages
### Requirements
* keras 2.0
* Scikit-Learn
* NumPy
* Pandas
* Tensorflow OR Theano

### MIMIC-III ###
Please apply for access to the publically available MIMIC-III DataBase via `https://www.physionet.org/`. 

### Instructions for Use ###

**Workflow**: MIMIC-III Access -> Database Buildding -> Preprocessing -> Pipeline

1. Once access to MIMIC-III files are granted, there are several options to build the database:
* Use the MIMIC-III repository online
* ... or go to `/preprocessing/DatabaseBuilder.py` and run this file to build the data tables from scratch. 
  1. Download the `.gz` or `.zip` files from _physionet_.
  2. Change the `path_` file from `DatabaseBuilder.py` to the appropriate folders containing the MIMIC-III zip files
  as well as the locations where the database is to be saved.  
  
2. Once the MIMIC-III data tables are constructed, run `preprocessing.py` to generate the feature representations of patients
and the labels for the clinical tasks. Follow instructions in README file under `/preprocessing/` folder.

3. Once feature and label files are ready, run `pipeline.py` on the appropriate settings and tasks to generate experimental results. Follow instructions in README file under `/pipeline` folder.

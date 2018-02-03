# Predictive Modeling in Urgent Care

This is the code repository for manuscript `Predictive Modeling in Urgent Care: A Comparative Study of Machine Learning Approaches` by *Fengyi Tang, Cao Xiao, Fei Wang, Jiayu Zhou*. Currently under revision. 

## Manuscript Abstract

**Objective**: The growing availability of rich clinical data such as patients' electronic health records (EHR) provide great opportunities to address a broad range of real-world questions in medicine. At the same time, artificial intelligence and machine learning based approaches have shown great premise on extracting insights from those data and helping with various clinical problems. The goal of this study is to conduct a systematic comparative study of different machine learning algorithms for several predictive modeling problems in urgent care. 

**Design**: We assess the performance of four benchmark prediction tasks (e.g., mortality and prediction, differential diagnostics and disease marker discovery) using medical histories, physiological time-series and demographics data from the Medical Information Mart for Intensive Care (MIMIC-III) database.

**Measurements**: For each given task, performance was estimated using standard measures including the area under the receiver operating characteristic (AUC) curve, F-1 score, sensitivity and specificity. Micro-averaged AUC was used for multi-class classification models.

**Results and Discussion**: Our results suggest that recurrent neural networks show the most promise in mortality prediction where temporal patterns in physiologic features alone can capture in-hospital mortality risk (AUC > 0.90). Temporal models did not provide additional benefit compared to deep models in differential diagnostics. When comparing the training-testing behaviors of readmission and mortality models, we illustrate that readmission risk may be independent of patient stability at discharge. We also introduce a multi-class prediction scheme for length of stay which preserves sensitivity and AUC with outliers of increasing duration despite decrease in sample size.

## Usages


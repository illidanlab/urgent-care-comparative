# Pipeline

This folder contains the experimental pipeline file. It calls upon the collapse models, LSTM, MLP and CNN modules from the `models` folder.

#Usage
##Requirements
* `keras 2.0`
* scikit-learn
* numpy
* Tensorflow OR Theano

## Procedure
1. Change `path_` files in `Sequential.py` and `Baselines.py`. 
2. Run `python Sequential.py` to train and test LSTM, CNN or MLP models. 
The default settings are 256 layers with non-stateful cells and no target-replication between time-steps. 
The default task is mortality prediction.
These can be changed with arg-parser arguments.
3. Run `python Baselines.py` to train and test classic models. Random Forest, Support Vector Machines and 
Logistic Regression Classifiers will simultaneously be trained and tested. 
4. The output for the pipeline files will be a data dictionary with AUC, f1-score, sensitivity and specificity 
for each k-fold split. The models trained will also be saved in the folder listed in `step 1`. 

## Arg Parser Options ##

_TBA_

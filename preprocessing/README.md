# Preprocessing MIMIC-III
`preprocessing.py` compiles a comprehensive `pandas` dataframe of features from MIMIC-III files. 

#Usages
##Requirements
* pandas 0.21+
* numpy

##Procedure
1. Change the `path_` files under `preprocessing.py`. Make sure that the pickle locations are correct for save files to be generated.
2. Run `python preprocessing.py`.

Output files should be:
_Label Files_
* A file called `y25` for differential diagnosis (multi-labels classification).
* A file called 'ylos_multinomial' for LOS prediction.
* A file called 'ym' for mortality prediction.
* A file called 'y_re' for readmission prediction.
_Feature Files_
* A file called 'x19' for 19 physiologic time series over last 48h for each admission.
* A file called 'h2v' for demographic history for each admission.
* A file called `dx' for diagnostic history (in group-code index) for each admission.

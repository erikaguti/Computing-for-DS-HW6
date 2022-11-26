# Computing-for-DS-HW6
Authers: Erika Gutierrez, Sam Minors

This is a library to predict whether some patients have diabetes mellitus. The premise is to provide a library of classes that other people can reuse and modify when writing their own models.

## load_data.py

Loads data from csv file and returns two dataframes, one for train and another for test. 

## preprocessor.py

Cleans the data by filling NaN with the mean value of the columns of height and weight. 

## Process ()

Classes to transform columns to features

## Model.py

ML process to train and predict. 

## sample_diabetes_mellitus_data.csv

Raw input data

## 


├── README.md          <- The top-level README for developers using this project.
┃  Diabetes_Mellitus_Predictor/
┃  ┣ Modeler/
┃  ┃ ┣ model.py
┃  ┃ ┗ __init__.py
┃  ┣ Prep/
┃  ┃ ┣ load_data.py
┃  ┃ ┣ preprocessor.py
┃  ┃ ┗ __init__.py
┃  ┣ Process/
┃  ┃ ┣ featureclass1.py
┃  ┃ ┣ featureclass2.py
┃  ┃ ┣ featureparent.py
┃  ┃ ┗ __init__.py
┃  ┣ README.md
┃  ┣ setup.py
┃  ┗ __init__.py
┃
┣ HW6-Instructions.pdf
┣ hw6.py
┣ sample_diabetes_mellitus_data.csv
┣

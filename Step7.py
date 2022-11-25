from Diabetes_Mellitus_Predictor.Prep.load_data import DataPrep
from Diabetes_Mellitus_Predictor.Prep.preprocessor import PreprocessRemove
from Diabetes_Mellitus_Predictor.Process.featureclass1 import Dummy
from Diabetes_Mellitus_Predictor.Process.featureclass2 import Interaction
from Diabetes_Mellitus_Predictor.Modeler.model import Model
import pandas as pd

data = DataPrep('sample_diabetes_mellitus_data.csv')

for col in data.dataset.columns:
    df = pd.DataFrame()
    if data.dataset[col].dtype == 'object':
        transform = Dummy().transform(data.dataset, col)
        df = pd.concat([df, transform])



#a. Create a class with a primary method that loads the data and returns two dataframes, one for train and another for test. Internally, the class can use the function defined in hw5.
#b. Create a preprocessor class that removes those rows that contain NaN values in the columns: age, gender, ethnicity.
#c. Create a preprocessor class that fills NaN with the mean value of the column in the columns: height, weight.
#d. Create at least two feature classes that transform some of the columns in the data set. These feature classes need to have the same structure defined by an abstract parent class (Remember: polymorphism).
#e. Create a model class with two primary methods: train and predict. When the model class is initialized, the constructor (init) should receive as inputs (at least):
#1. Feature columns that are going to be used
#2. Target column that is going to be used
#3. (bonus) Hyperparameters of the model to be used.
#f. The model class should have as private attributes each of the inputs of the constructor and an additional one, called “model” that will be a model from sklearn chosen by the team (such as LogisticRegression or RandomForestClassifier) as a public attribute of the class.
#1. The train method should receive the train data containing at least the feature and target columns defined and fit the self.model on the train data using the features and the target (to filter columns) passed when the class is initialize, and return nothing. 2.The predict method should receive a dataframe, use the features passed to filter the columns and return the predicted probabilities using the .predict_proba method of the sklearn class selected.
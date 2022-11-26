from Diabetes_Mellitus_Predictor.Prep.load_data import DataPrep
from Diabetes_Mellitus_Predictor.Prep.preprocessor import PreprocessRemove
from Diabetes_Mellitus_Predictor.Prep.preprocessor import PreprocessFill
from Diabetes_Mellitus_Predictor.Process.featureclass1 import Dummy
from Diabetes_Mellitus_Predictor.Process.featureclass2 import Interaction
from Diabetes_Mellitus_Predictor.Modeler.model import Model
import pandas as pd
from sklearn.metrics import roc_auc_score

# Step 1: Load initial dataset
prep = DataPrep('sample_diabetes_mellitus_data.csv')
prep.dataset.drop('Unnamed: 0', axis=1, inplace=True)

# Step 2: Remove nulls in age, gender, and ethnicity

prep.dataset = PreprocessRemove().removenullrows(prep.dataset, ['age', 'gender', 'ethnicity'])

# Step 3: Fill nulls with with average of the column for height and weight

prep.dataset = PreprocessFill().fillnulls(prep.dataset, ['height', 'weight'])

# Step 4: Feature Engineering:

## Step 4.1: Create dummy columns for all string columns in the dataset

tdfs = []
for col in prep.dataset.select_dtypes('object').columns:
        transform = Dummy().transform(prep.dataset, col)
        tdfs.append(transform)

prep.dataset = pd.concat([pd.concat(tdfs, axis = 1), prep.dataset], axis = 1)

## Step 4.2: Create interaction term columns for all numerical columns in the dataset
tdfs = []
for col in prep.dataset.select_dtypes('number').columns[:-1]:
    transform = Interaction().transform(prep.dataset, col)
    tdfs.append(transform)

prep.dataset = pd.concat([pd.concat(tdfs, axis=1), prep.dataset], axis = 1)


# Step 5: Prepare train and test datasets
prep.dataset.dropna(axis = 1, inplace = True)

prep.dataset = prep.dataset.select_dtypes('number')

prep.independent_variables = prep.dataset.columns[:-1]

train, test = prep.get_train_test_datasets(prep.dataset[prep.independent_variables], prep.dataset[prep.target_variable])


# Step 6: Run model
prep.dataset.dropna(axis = 1, inplace = True)

prep.dataset = prep.dataset.select_dtypes('number')

prep.independent_variables = prep.dataset.columns[:-1]

train, test = prep.get_train_test_datasets(prep.dataset[prep.independent_variables], prep.dataset[prep.target_variable])

diabetes_predictor = Model(prep.independent_variables, prep.target_variable)

diabetes_predictor.train(train)

probas = diabetes_predictor.predict(test)

test['predictions'] = probas[:,1]

print('Area Under the Receiver Operating Characteristic Curve = ', roc_auc_score(test['diabetes_mellitus'], test['predictions']))




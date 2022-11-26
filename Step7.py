from Diabetes_Mellitus_Predictor.Prep.load_data import DataPrep
from Diabetes_Mellitus_Predictor.Prep.preprocessor import PreprocessRemove
from Diabetes_Mellitus_Predictor.Prep.preprocessor import PreprocessFill
from Diabetes_Mellitus_Predictor.Process.featureclass1 import Dummy
from Diabetes_Mellitus_Predictor.Process.featureclass2 import Interaction
from Diabetes_Mellitus_Predictor.Modeler.model import Model
import pandas as pd
from scipy.stats import pearsonr


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


# Step 5: Prepare model


## 5.1 compute Pearson correlations to remove redundant features
prep.dataset.dropna(axis = 1, inplace = True)
correlations = {}
for col in prep.dataset.columns:
    try:
        correlations.update({col:pearsonr(prep.dataset[col], prep.dataset['diabetes_mellitus'])})
    except Exception:
        correlations.update({col:0})

for num in correlations.values():
    try:
        if round(num[0], 1) > .5:
            print("corr", num[0])
    except:
        print(num)
    

# Step 6: Run model



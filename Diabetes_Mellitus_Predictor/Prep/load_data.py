from sklearn.model_selection import train_test_split
import pandas as pd

class DataPrep:
    def __init__(self, datapath):
        self.dataset = pd.read_csv(datapath)
        self.independent_variables = list(self.dataset.columns[:-1])
        self.target_variable = list(self.dataset.columns[-1:])
    
    def get_train_test_datasets(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        training = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        return training, test



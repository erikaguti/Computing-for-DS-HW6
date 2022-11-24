from sklearn.model_selection import train_test_split
import pandas as pd

class DataPrep:
    def __init__(self, datapath, indep, dep):
        self.dataset = pd.read_csv(datapath)
        self.independent_variables = indep
        self.target_variable = dep
    
    def get_features(self):
        X = self.dataset.loc[:, [self.independent_variables]]
        return X
    def get_target(self):
        y = self.dataset.loc[:,[self.target_variable]]
        return y
    def get_train_test_datasets(self, X, y):
        X_train, y_train, X_test, y_test = train_test_split(X, y)
        training = pd.DataFrame(X_train, columns = [self.independent_variables])
        training[f'{self.target_variable}'] = y_train
        test = pd.DataFrame(X_test, columns=self.independent_variables)
        test[f'{self.target_variable}'] = y_test
        return training, test





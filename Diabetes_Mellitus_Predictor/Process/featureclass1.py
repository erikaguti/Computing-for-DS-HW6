from Diabetes_Mellitus_Predictor.Process.featureparent import Feature
import pandas as pd

class Dummy(Feature):
    def transform(self, df, column):
        return pd.get_dummies(df[column], prefix=column)
        
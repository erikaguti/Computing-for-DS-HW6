from featureparent import Feature
import pandas as pd

class Dummy(Feature):
    def transform(df, column):
        return pd.get_dummies(df[column])

    def apply(transformeddf, originaldf):
        for col in transformeddf.columns:
            originaldf[f'{col}'] = transformeddf[f'{col}']
        return originaldf
        



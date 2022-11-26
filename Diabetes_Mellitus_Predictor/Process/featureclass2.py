from Diabetes_Mellitus_Predictor.Process.featureparent import Feature
import pandas as pd


class Interaction(Feature):
    def transform(self, df, column):
        interactions = {}
        for col in df.select_dtypes('number').columns:
            if col not in [column, 'diabetes_mellitus']:
                interactions.update({column+"_x_"+col: list(df[column]*df[col])})
        return pd.DataFrame.from_dict(interactions)









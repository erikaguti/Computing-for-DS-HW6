from featureparent import Feature
import pandas as pd


class Interaction(Feature):
    def transform(self, df, column):
        interactions = {}
        for col in df.columns[1:-1]:
            if column != col:
                if df[col].dtype == df[column].dtype:
                    try:
                        interactions.update({column+"_x_"+col: list(df[column]*df[col])})
                    except TypeError:
                        interactions.update({column+"_x_"+col: list(df[column]+df[col])})
        return pd.DataFrame.from_dict(interactions)

    def apply(self, transformeddf, originaldf):
        for col in transformeddf.columns:
            originaldf[f'{col}'] = transformeddf[f'{col}']
        return originaldf








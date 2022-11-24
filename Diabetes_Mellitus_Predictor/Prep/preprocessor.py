import numpy as np

class PreprocessRemove:
    def removenullrows(self, df, columns):
        for col in columns:
            df.dropna(col, inplace = True)
        return df

class PreprocessFill:
    def fillnulls(self, df, columns):
        for col in columns:
            df[col].fillna(df[col].mean(), inplace = True)
        return df

from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Model:
    def __init__(self, features, target):
        self._features = features
        self._target = target
        self.model = RandomForestClassifier()
    def train(self, data):
        self.model.fit(data[self._features], data[self._target].to_numpy().ravel())
        return 
    def predict(self, data):
        return self.model.predict_proba(data[self._features])



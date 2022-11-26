from abc import ABC, abstractmethod

class Feature(ABC):
    @abstractmethod
    def transform(self):
        pass


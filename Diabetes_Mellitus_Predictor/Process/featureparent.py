from abc import ABCMeta, abstractmethod

class Feature:
    @abstractmethod
    def prep():
        pass
    @abstractmethod
    def transformation():
        pass
    @abstractmethod
    def apply():
        pass

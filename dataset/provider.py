from abc import ABC, abstractmethod
import yaml

class dataProvider():
    @abstractmethod
    def provideData(self, params):
        pass

    @abstractmethod
    def provideLabel(self, model_name, task_name=None):
        pass
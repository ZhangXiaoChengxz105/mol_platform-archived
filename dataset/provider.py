from abc import ABC, abstractmethod
import yaml

class dataProvider():
    @abstractmethod
    def provideData(self, params):
        pass
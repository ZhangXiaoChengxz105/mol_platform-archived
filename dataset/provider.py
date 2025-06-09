from abc import ABC, abstractmethod

class dataProvider():
    @abstractmethod
    def provideData(self, params):
        pass
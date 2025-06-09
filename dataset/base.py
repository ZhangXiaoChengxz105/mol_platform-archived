from abc import ABC, abstractmethod

class BaseDataset:
    def __init__(self, datasetName: str, datasetPath: str):
        self.datasetName = datasetName
        self.datasetPath = datasetPath
        self.data = None  

    @abstractmethod
    def loadData(self):
        raise NotImplementedError("loadData()")
    
    @abstractmethod
    def preprocessData(self):
        raise NotImplementedError("preprocessData()")


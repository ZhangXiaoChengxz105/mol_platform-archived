from base import BaseDataset
from provider import dataProvider
import pandas as pd

class sequenceDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    def provideData(self, params):
        smiles_list = self.data['smiles'].tolist()
        labels = self.data['p_np'].tolist()
        return smiles_list, labels
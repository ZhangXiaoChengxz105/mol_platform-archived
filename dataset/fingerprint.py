from base import BaseDataset
from provider import dataProvider
# from dataset.utils import smiles_to_mol
import pandas as pd

class fingerprintDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    def provideData(self, params):
        smiles_list = self.data['smiles'].tolist()
        labels = self.data['p_np'].tolist()
        return smiles_list, labels

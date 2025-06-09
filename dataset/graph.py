from base import BaseDataset
from provider import dataProvider
# from torch_geometric.data import Data
import pandas as pd

class graphDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    def provideData(self, params=None):
        smiles_list = self.data['smiles'].tolist()
        labels_list = self.data['p_np'].tolist()
        return smiles_list, labels_list
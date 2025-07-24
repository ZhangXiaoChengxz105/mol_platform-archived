from rdkit.Chem import MACCSkeys
from rdkit import Chem
from base import BaseDataset
from provider import dataProvider
import pandas as pd
from rdkit.Chem import AllChem
import yaml
import numpy as np
import torch

class ECFPDataset(BaseDataset, dataProvider):
    def loadData(self):
        self.data = pd.read_csv(self.datasetPath)

    def preprocessData(self):
        pass

    @staticmethod
    def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.tensor(arr, dtype=torch.float32)

    def provideData(self, model_name):
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        inputs, labels = [], []
        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])
            fp = self.smiles_to_ecfp(smiles)
            if fp is not None:
                inputs.append(fp)
                labels.append(label)

        return {"input": torch.stack(inputs), "label": labels}

    def provideLabel(self, model_name, task_name=None):
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        labels = []
        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            if self.smiles_to_ecfp(smiles) is not None:
                label = row[task_name] if task_name else (row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols]))
                labels.append(label)
        return labels


class MACCSDataset(BaseDataset, dataProvider):
    def loadData(self):
        self.data = pd.read_csv(self.datasetPath)

    def preprocessData(self):
        pass

    @staticmethod
    def smiles_to_maccs(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.tensor(arr, dtype=torch.float32)

    def provideData(self, model_name):
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        inputs, labels = [], []
        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])
            fp = self.smiles_to_maccs(smiles)
            if fp is not None:
                inputs.append(fp)
                labels.append(label)

        return {"input": torch.stack(inputs), "label": labels}

    def provideLabel(self, model_name, task_name=None):
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        labels = []
        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            if self.smiles_to_maccs(smiles) is not None:
                label = row[task_name] if task_name else (row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols]))
                labels.append(label)
        return labels

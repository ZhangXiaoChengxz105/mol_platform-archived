from base import BaseDataset
from provider import dataProvider
import pandas as pd
from rdkit.Chem import AllChem
import numpy as np
import torch
from rdkit import Chem
import yaml

class fingerprintDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass


    @staticmethod
    def smiles_to_fingerprint(smiles, radius=2, n_bits=2048, as_tensor=True):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        if as_tensor:
            return torch.tensor(arr, dtype=torch.float32)
        return arr


    def provideData(self, model_name):
        """
        Returns:
            dict:
                {
                    "input": torch.Tensor [N, 2048],    # 每个 SMILES 转换后的 ECFP 指纹向量
                    "label": List[int | List[float]]   # 对应标签列表
                }
        """
        with open("smile_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        fingerprint_list = []
        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprint_list.append(fp)
                label_list.append(label)

        return {
            "input": torch.stack(fingerprint_list),  # Tensor shape: [N, 2048]
            "label": label_list
        }
    def provideLabel(self, model_name, task_name=None):
        with open("smile_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        if task_name and task_name not in label_cols:
            raise ValueError(f"task_name '{task_name}' not found in label columns: {label_cols}")

        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                if task_name:
                    label = row[task_name]
                else:
                    label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])
                label_list.append(label)

        return label_list

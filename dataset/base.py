from abc import ABC, abstractmethod
from rdkit import Chem
import yaml
import os

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

    def provideSmilesAndLabel(self, model_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "smile_config.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)


        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        smiles_list = []
        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            if Chem.MolFromSmiles(smiles) is not None:
                smiles_list.append(smiles)
                label_list.append(label)

        return {
            "smiles": smiles_list,
            "label": label_list
        }
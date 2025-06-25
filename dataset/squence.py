from base import BaseDataset
from provider import dataProvider
import torch
from rdkit import Chem
from torch_geometric.data import Data
import pandas as pd
import yaml

class sequenceDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    @staticmethod
    def smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            # 原子类型：确保在0-118范围内
            atomic_num = atom.GetAtomicNum()
            if atomic_num >= 119:
                atomic_num = 0  # 使用0作为unknown token
            
            # 手性信息：确保在0-2范围内  
            try:
                chiral_tag = int(atom.GetChiralTag())
                if chiral_tag >= 3:
                    chiral_tag = 0
            except:
                chiral_tag = 0
            
            features = [atomic_num, chiral_tag]
            atom_features.append(features)
    
        # 边特征处理
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = int(bond.GetBondType())
            bond_dir = int(bond.GetBondDir())
        
            # 确保键类型和方向在有效范围内
            if bond_type >= 5: bond_type = 0
            if bond_dir >= 3: bond_dir = 0
        
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[bond_type, bond_dir], [bond_type, bond_dir]])
    
        return Data(
            x=torch.tensor(atom_features, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.long),
            y=None
        )

    def provideData(self, model_name):
        """
        Args:
            model_name (str): The name of the dataset model, must match a key in smile_config.yaml.

        Returns:
            dict:
                {
                    "smiles_list": List[str],                   # List of valid SMILES strings
                    "true_label_list": List[int | List[float]]  # Corresponding labels (single or multi-label)
                }
        """
        with open("smile_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        smiles_list = []
        true_label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            if self.smiles_to_graph(smiles) is not None:
                smiles_list.append(smiles)
                true_label_list.append(label)

        return {
            "smiles_list": smiles_list,
            "true_label_list": true_label_list
        }

    def provideLabel(self, model_name, task_name=None):
        """
           Args:
               model_name (str): Dataset name, must match a key in smile_config.yaml.
               task_name (str, optional): Specific label column to extract. If None, all label columns are returned.

           Returns:
               List[int | float | List[float]]:
                   A list of label values (single-label or multi-label vector) corresponding to valid SMILES entries.
           """
        import yaml

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

            if self.smiles_to_graph(smiles) is not None:
                if task_name:
                    label = row[task_name]
                else:
                    label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])
                label_list.append(label)

        return label_list


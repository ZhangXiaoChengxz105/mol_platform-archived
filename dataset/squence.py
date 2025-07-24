from base import BaseDataset
from provider import dataProvider
import torch
import pandas as pd
import yaml

class sequenceDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    @staticmethod
    def smiles_to_sequence(smiles, vocab=None, max_len=128):
        if smiles is None:
            return None

        if vocab is None:
            charset = sorted(list(set(smiles)))
            vocab = {ch: i + 1 for i, ch in enumerate(charset)}

        sequence = [vocab.get(ch, 0) for ch in smiles]
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence += [0] * (max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long)

    def provideData(self, model_name):
        """
        Returns:
            dict: {
                "input": torch.Tensor [N, max_len],    # SMILES 序列转化后的张量
                "label": List[int or List[float]]      # 对应的标签
            }
        """
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        vocab = self.build_vocab(smiles_col)
        sequence_list = []
        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            seq = self.smiles_to_sequence(smiles, vocab)
            if seq is not None:
                sequence_list.append(seq)
                label_list.append(label)

        return {
            "input": torch.stack(sequence_list),  # shape [N, max_len]
            "label": label_list
        }

    def build_vocab(self, smiles_col):
        charset = set()
        for s in self.data[smiles_col]:
            if isinstance(s, str):
                charset.update(list(s))
        vocab = {ch: i + 1 for i, ch in enumerate(sorted(charset))}  # 0 = padding
        return vocab

    def provideLabel(self, model_name, task_name=None):
        with open("moleculnet_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if model_name not in config["datasets"]:
            raise ValueError(f"No such config for model: {model_name}")

        task_cfg = config["datasets"][model_name]
        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        if task_name and task_name not in label_cols:
            raise ValueError(f"task_name '{task_name}' not in label columns: {label_cols}")

        vocab = self.build_vocab(smiles_col)
        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            seq = self.smiles_to_sequence(smiles, vocab)
            if seq is not None:
                if task_name:
                    label = row[task_name]
                else:
                    label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])
                label_list.append(label)

        return label_list



# 基类
from abc import ABC, abstractmethod
import yaml
class BaseModel(ABC):
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model = None

    @abstractmethod
    def featurize(self, smiles):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


# 分子指纹+机器学习
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem

class FingerprintModel(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.fp_type = self.config['fingerprint_type']
        self.fp_size = self.config['fingerprint_size']

    def featurize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if self.fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_size)
        elif self.fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError("Unsupported fingerprint type")
        return list(fp)

    def build_model(self):
        if self.config['task_type'] == 'classification':
            return RandomForestClassifier(**self.config['model_params'])
        else:
            return RandomForestRegressor(**self.config['model_params'])

    def train(self, X, y):
        self.model = self.build_model()
        self.model.fit(X, y)

    def predict(self, X):
        if self.config['task_type'] == 'classification':
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)



# 字符串+序列模型
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels, char_dict, max_length=100):
        self.sequences = [self.encode(s, char_dict, max_length) for s in smiles_list]
        self.labels = labels

    def encode(self, smile, char_dict, max_length):
        return [char_dict.get(c, 0) for c in smile.ljust(max_length)[:max_length]]

class SequenceModel(BaseModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.char_dict = {c:i for i,c in enumerate(self.config['charset'])}
        self.max_length = self.config['max_length']

    def featurize(self, smiles):
        return [self.char_dict.get(c, 0) for c in smiles.ljust(self.max_length)[:self.max_length]]

    def build_model(self):
        return LSTMModel(
            vocab_size=len(self.char_dict),
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        )

    def train(self, X, y):
        self.model = self.build_model()
        # 实际训练逻辑需补充，这里仅示意
        pass

    def predict(self, X):
        # 实际预测逻辑需补充
        pass

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))



# 分子图+图神经网络
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data

class GraphModel(BaseModel):
    def featurize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_features = self._get_atom_features(mol)
        edge_index = self._get_edge_index(mol)
        return Data(x=atom_features, edge_index=edge_index)

    def _get_atom_features(self, mol):
        return torch.stack([self._atom_feature(a) for a in mol.GetAtoms()])

    def _atom_feature(self, atom):
        return torch.tensor([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge()
        ], dtype=torch.float)

    def _get_edge_index(self, mol):
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])  # 无向图
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_model(self):
        return GNNModel(
            node_dim=3,  # 根据实际特征数调整
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        )

    def train(self, X, y):
        self.model = self.build_model()
        # 实际训练逻辑需补充
        pass

    def predict(self, X):
        # 实际预测逻辑需补充
        pass

class GNNModel(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(node_dim, hidden_dim)
        self.conv2 = tg.nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x.mean(dim=0))



# apiclass ModelRunner:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.model = self._init_model()

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_model(self):
        model_type = self.config['model_type']
        if model_type == 'fingerprint':
            return FingerprintModel(self.config)
        elif model_type == 'sequence':
            return SequenceModel(self.config)
        elif model_type == 'graph':
            return GraphModel(self.config)

    def run(self, data):
        features = [self.model.featurize(smiles) for smiles in data['smiles']]
        self.model.train(features, data['labels'])
        return self.model.predict(features)
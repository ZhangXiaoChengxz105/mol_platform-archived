# models.py
import torch
import numpy as np
import torch_geometric as tg
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier

class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass

class FingerprintModel(base_model):
    def __init__(self, name, path, fingerprint_type='morgan', fingerprint_size=2048):
        super().__init__(name, path)
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.model = None

    def featurize(self, smiles_list):
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if self.fingerprint_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fingerprint_size)
            elif self.fingerprint_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError("Unsupported fingerprint type")
            features.append(np.array(fp))
        return np.array(features)

    def load_weights(self, path):
        import joblib
        self.model = joblib.load(path)

    def train(self, X, y):
        if self.model is None:
            self.model = RandomForestClassifier()
        self.model.fit(X, y)
        # 可选：保存模型权重
        import joblib
        joblib.dump(self.model, f"{self.path}/fingerprint_model.joblib")

    def predict(self, data):
        features = self.featurize(data)
        return self.model.predict_proba(features)[:, 1]  # 假设二分类

class SequenceModel(base_model):
    def __init__(self, name, path, pretrained_model="ibm/MOLFORMER-XL", max_length=100):
        super().__init__(name, path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.max_length = max_length
        self.proj = nn.Linear(768, 1)  # 假设回归任务

    def featurize(self, smiles_list):
        return self.tokenizer(smiles_list, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def load_weights(self, path):
        # 自定义权重加载，如微调后的模型
        self.model = AutoModel.from_pretrained(path)
        self.proj.load_state_dict(torch.load(f"{path}/proj.pth"))

    def train(self, X, y, epochs=3, batch_size=32):
        # 假设X是SMILES列表，y是标签
        from torch.utils.data import Dataset, DataLoader
        class SmilesDataset(Dataset):
            def __init__(self, smiles, labels):
                self.smiles = smiles
                self.labels = labels
            def __len__(self):
                return len(self.smiles)
            def __getitem__(self, idx):
                return self.smiles[idx], self.labels[idx]
        dataset = SmilesDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.proj.parameters()), lr=1e-4)
        criterion = nn.MSELoss()  # 回归任务
        for epoch in range(epochs):
            for batch_smiles, batch_labels in loader:
                inputs = self.featurize(batch_smiles)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
                preds = self.proj(features).squeeze()
                loss = criterion(preds, torch.tensor(batch_labels, dtype=torch.float))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 可选：保存模型权重
        self.model.save_pretrained(f"{self.path}/sequence_model")
        torch.save(self.proj.state_dict(), f"{self.path}/sequence_model/proj.pth")

    def predict(self, data):
        inputs = self.featurize(data)
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            preds = self.proj(features).squeeze()
        return preds.numpy()

class GINModel(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = tg.nn.GINConv(
            nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        self.conv2 = tg.nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return self.fc(x.mean(dim=0))

class GraphModel(base_model):
    def __init__(self, name, path, node_dim=3, hidden_dim=128, output_dim=1):
        super().__init__(name, path)
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = GINModel(node_dim, hidden_dim, output_dim)

    def _mol_to_graph(self, mol):
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge()
            ]
            atom_features.append(features)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        return Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

    def featurize(self, smiles_list):
        graphs = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            graphs.append(self._mol_to_graph(mol))
        return graphs

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, X, y, epochs=10, batch_size=32):
        # 假设X是SMILES列表，y是标签
        graphs = self.featurize(X)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i+batch_size]
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float)
                batch_preds = []
                for g in batch_graphs:
                    pred = self.model(g.x, g.edge_index)
                    batch_preds.append(pred)
                preds = torch.stack(batch_preds).squeeze()
                loss = criterion(preds, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 可选：保存模型权重
        torch.save(self.model.state_dict(), f"{self.path}/graph_model.pth")

    def predict(self, data):
        graphs = self.featurize(data)
        preds = []
        with torch.no_grad():
            for g in graphs:
                pred = self.model(g.x, g.edge_index)
                preds.append(pred.item())
        return np.array(preds)



# api
from abc import ABC, abstractmethod

class model_runner_interface(ABC):
    @abstractmethod
    def run(self, data):
        pass

class FingerprintRunner(model_runner_interface):
    def __init__(self, model):
        self.model = model
    def run(self, data):
        return self.model.predict(data)

class SequenceRunner(model_runner_interface):
    def __init__(self, model):
        self.model = model
    def run(self, data):
        return self.model.predict(data)

class GraphRunner(model_runner_interface):
    def __init__(self, model):
        self.model = model
    def run(self, data):
        return self.model.predict(data)


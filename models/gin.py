import torch
import torch.nn as nn
import numpy as np
import torch_geometric as tg
from torch_geometric.data import Data
from rdkit import Chem
from models import base_model

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

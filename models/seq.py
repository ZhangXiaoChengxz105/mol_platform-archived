import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from models import base_model


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

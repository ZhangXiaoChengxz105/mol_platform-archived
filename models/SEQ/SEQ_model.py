import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from base_model import base_model
'''
class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.task = task(name)
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass
'''
# models/SEQ/SEQ_model.py
from SEQ_data import TOKENIZER
class SEQ(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        # SEQ细分task
        if self.name in ["Tox21","ClinTox","MUV","SIDER"]:
            self.task = "classification_multitask"
        # 初始化tokenizer
        self.tokenizer = TOKENIZER
        self.model = self.Net()
        self.tok_emb = nn.Embedding(len(self.tokenizer.vocab), 768)

    def load_weights(self, path=None):
        load_path = path if path is not None else self.path
        tokemb_path = load_path.replace(".pth","_tokemb.pth")
        if not load_path:
            raise ValueError("必须提供模型路径（path参数或初始化时的path）")        
        if load_path.endswith(".pth"):
            try:
                # 初始化模型（参数与pubchem完全一致）
                combined = torch.load(load_path)
                self.model.load_state_dict(combined["model"])
                self.tok_emb.load_state_dict(combined["tok_emb"])
                print("load Net success!\n\n")
            except Exception as e:
                raise RuntimeError(f"权重加载失败: {str(e)}")
        else:
            print("invalid path!")

    def predict(self, data):
        if not self.model:
            raise RuntimeError("请先调用load_weights()")
        # 模拟 training_step 和 validation_step 中的处理流程
        idx = data[0]  # 假设第一个张量是输入的 token ids
        mask = data[1]
        token_embeddings = self.tok_emb(idx)  # 词嵌入
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        smiles_emb = sum_embeddings / sum_mask

        self.model.eval()
        with torch.no_grad():
            return self.model(smiles_emb)

    class Net(nn.Module):
            smiles_embed_dim = 768
            dims = [768,768,768,1]


            def __init__(self, smiles_embed_dim = smiles_embed_dim, dims=dims, dropout=0.2):
                super().__init__()
                self.desc_skip_connection = True 
                self.fcs = []  # nn.ModuleList()
                print('dropout is {}'.format(dropout))

                self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
                self.dropout1 = nn.Dropout(dropout)
                self.relu1 = nn.GELU()
                self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
                self.dropout2 = nn.Dropout(dropout)
                self.relu2 = nn.GELU()
                self.final = nn.Linear(smiles_embed_dim, 1)

            def forward(self, smiles_emb):
                x_out = self.fc1(smiles_emb)
                x_out = self.dropout1(x_out)
                x_out = self.relu1(x_out)

                if self.desc_skip_connection is True:
                    x_out = x_out + smiles_emb

                z = self.fc2(x_out)
                z = self.dropout2(z)
                z = self.relu2(z)
                if self.desc_skip_connection is True:
                    z = self.final(z + x_out)
                else:
                    z = self.final(z)

                return z
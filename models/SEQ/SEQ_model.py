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
from check_utils import get_datasets_measure_numbers
from SEQ_data import TOKENIZER

from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from fast_transformers.masking import LengthMask as LM


class SEQ(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        # SEQ细分task, 初始化模型
        if self.name in ["Tox21","ClinTox","MUV","SIDER"]:
            self.task = "classification_multitask"
            self.model = self.Net(dims = get_datasets_measure_numbers(name))
        
        elif self.task == "classification":
            self.model = self.Net(dims = 2)
        else:
            self.model = self.Net()
        # 初始化tokenizer

    def load_weights(self, path=None):
        load_path = path if path is not None else self.path
        tokemb_path = load_path.replace(".pth","_tokemb.pth")
        if not load_path:
            raise ValueError("必须提供模型路径(path参数或初始化时的path)")        
        if load_path.endswith(".pth"):
            try:
                # 初始化模型（参数与pubchem完全一致）
                state_dict = torch.load(load_path)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print("load Net success!\n\n")
            except Exception as e:
                raise RuntimeError(f"权重加载失败: {str(e)}")
        else:
            print("invalid path!")

    def predict(self, data):
        if not self.model:
            raise RuntimeError("请先调用load_weights()")
        # 模拟 training_step 和 validation_step 中的处理流程
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    

    class Net(nn.Module):
            smiles_embed_dim = 768

            def __init__(self, smiles_embed_dim = smiles_embed_dim, dims=1, dropout=0.2):
                super().__init__()
                self.desc_skip_connection = True 
                self.fcs = []  # nn.ModuleList()
                # print('dropout is {}'.format(dropout))
                # embedding layer
                self.tokenizer = TOKENIZER
                self.tok_emb = nn.Embedding(len(self.tokenizer.vocab), 768)
                # transform layer
                torch.manual_seed(42)  # set deterministic seed
                builder = rotate_builder.from_kwargs(
                    n_layers=12,
                    n_heads=12,
                    query_dimensions=768//12,
                    value_dimensions=768//12,
                    feed_forward_dimensions=768,
                    attention_type='linear',
                    feature_map=partial(GeneralizedRandomFeatures, n_dims=32),
                    activation='gelu',
                    )
                self.blocks = builder.get()


                self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
                self.dropout1 = nn.Dropout(dropout)
                self.relu1 = nn.GELU()
                self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
                self.dropout2 = nn.Dropout(dropout)
                self.relu2 = nn.GELU()
                self.final = nn.Linear(smiles_embed_dim, dims)

            def forward(self, data_list):
                # 提取所有样本的输入和掩码
                idx_list = [data[0] for data in data_list]
                mask_list = [data[1] for data in data_list]
                
                # 检查是否有样本
                if not idx_list:
                    return torch.tensor([])
                
                # 确保所有样本在同一设备上
                device = idx_list[0].device
                
                # 对序列进行填充（如果需要）
                max_len = max([idx.size(1) for idx in idx_list])
                batch_size = len(idx_list)
                
                # 创建批量输入张量
                idx_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
                mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
                
                for i, (idx, mask) in enumerate(zip(idx_list, mask_list)):
                    seq_len = idx.size(1)
                    idx_batch[i, :seq_len] = idx
                    mask_batch[i, :seq_len] = mask
                
                # 统一处理所有样本
                token_embeddings = self.tok_emb(idx_batch)  # 词嵌入
                token_embeddings = self.blocks(token_embeddings, length_mask=LM(mask_batch.sum(-1)))
                
                input_mask_expanded = mask_batch.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                smiles_emb = sum_embeddings / sum_mask

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
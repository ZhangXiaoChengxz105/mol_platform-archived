import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import base_model
from check_utils import get_datasets_measure_numbers

class RNNModel(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        self.char_to_index = {}  # 词汇表映射
        self.normalizer = None
        self.model = None
        
        # 模型参数
        self.embed_dim = 128
        self.hidden_size = 512  # 与论文一致
        self.num_layers = 1     # 与论文一致
        self.dropout = 0.2

    def load_weights(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"模型文件不存在: {self.path}")
        
        try:
            checkpoint = torch.load(self.path, map_location='cpu')
            
            # 加载关键参数
            self.char_to_index = checkpoint['char_to_index']  # 加载词汇表
            output_dim = checkpoint['output_dim']
            
            # 构建模型
            self.model = self.Net(
                vocab_size=len(self.char_to_index) + 1,  # +1 for padding
                embed_dim=self.embed_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_dim=output_dim,
                dropout=self.dropout
            )
            
            # 加载权重
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            print(f"RNN模型权重已从 {self.path} 加载")
            
            # 加载归一化器
            if 'normalizer' in checkpoint and checkpoint['normalizer']:
                self.normalizer = Normalizer(torch.tensor([0.0]))
                self.normalizer.load_state_dict(checkpoint['normalizer'])
        except Exception as e:
            print(f"详细错误信息: {str(e)}")
            raise RuntimeError(f"权重加载失败: {str(e)}")

    def predict(self, sequences, lengths):
        if self.model is None:
            raise RuntimeError("请先调用load_weights()加载模型权重")
            
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequences, lengths)
            
            if self.task == "classification":
                predictions = torch.sigmoid(predictions)
            elif self.task == "regression" and self.normalizer:
                predictions = self.normalizer.denorm(predictions)
                
            return predictions

    class Net(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, output_dim, dropout=0.2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.rnn = nn.GRU(
                embed_dim, hidden_size, num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            # 按照论文的三层全连接结构
            self.fc1 = nn.Linear(hidden_size, 128)
            self.fc2 = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, output_dim)
            
            # Xavier初始化权重
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
        
        def forward(self, sequences, lengths):
            embedded = self.embedding(sequences)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
            last_hidden = hidden[-1]
            
            # 三层全连接
            x = torch.relu(self.fc1(last_hidden))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

class Normalizer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
    
    def norm(self, tensor):
        return (tensor - self.mean) / self.std
    
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    
    def state_dict(self):
        return {'mean': self.mean.item(), 'std': self.std.item()}
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
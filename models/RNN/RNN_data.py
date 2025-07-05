import torch
import numpy as np

def smiles_to_tensor(smiles_list, char_to_index, max_length=100):
    """转换SMILES为模型输入张量"""
    sequences, lengths = [], []
    for smile in smiles_list:
        # 过滤无效字符
        seq = [char_to_index.get(c, 0) for c in smile][:max_length]
        sequences.append(seq)
        lengths.append(len(seq))
        
    # 填充序列
    max_len = max(lengths) if lengths else 0
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        
    return torch.tensor(padded), torch.tensor(lengths)
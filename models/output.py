from data_transformer import smiles_to_graph
from GNN import GNN

def gnn_predict(name, path, smiles):
    """
    简单的GNN预测接口
    Args:
        name: 属性名（如'BBBP'）
        path: 路径名 （如 models/finetune_gin/BBBP_best.pth）
        smiles: 分子SMILES字符串
    Returns:
        预测结果
    """
    
    # 创建模型
    model = GNN(name, path)
    model.load_weights(path)
    graph_data = smiles_to_graph(smiles)
    pred = model.predict(graph_data)
    
    return pred

# 测试
result = gnn_predict("BBBP", "models/finetune_gin/BBBP_best.pth","c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O")
print(result)

# validation 
import torch
print(torch.softmax(result, dim=1)) # should be close to [0, 1]


    
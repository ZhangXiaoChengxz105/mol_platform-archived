from data_transformer import smiles_to_graph
from GNN import GNN
import torch

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
    pred_logits = model.predict(graph_data)
    pred_prop = torch.softmax(pred_logits, dim=1)[0][1]
    if model.task == "classification":
        pred_lable = pred_prop > 0.5
    else:
        pred_lable = None
    return smiles, model.task, pred_prop.item(), pred_lable.item()

# 测试
smiles, task, pred_logits, result = gnn_predict("BBBP", "models/finetune_gin/BBBP_best.pth","c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O")
print("Output test:")
print(smiles)
print(task) # task
print(pred_logits) # pred_prop
print(result) # pred_lable




    
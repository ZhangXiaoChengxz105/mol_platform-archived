import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GNN_data import smiles_to_graph
from GNN_model import GNN
from check_utils import get_datasets_measure_names, validate_datasets_measure_names

def gnn_predict(name, target, smiles):
    """
    简单的GNN预测接口
    Args:
        name: 属性名（如'BBBP'）
        path: 路径名 （如 models/finetune_gin/BBBP_best.pth）
        smiles: 分子SMILES字符串
    Returns:
        预测结果
    """
    validate_datasets_measure_names(name, target)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "GNN_finetune", f"{name}_{target}.pth")
    # 创建模型
    model = GNN(name)
    model.load_weights(path)
    graph_data = smiles_to_graph(smiles)
    pred_logits = model.predict(graph_data)
    result = {
        "smiles": smiles,
        "task": model.task,
        "prediction": None,
        "label": None,
        "confidence": None
    }
    if model.task == "classification":
        # 分类任务处理
        pred_probs = torch.softmax(pred_logits, dim=1)
        result["prediction"] = pred_probs[0][1].item()  # 正类概率
        result["label"] = 1 if result["prediction"] > 0.5 else 0
    elif model.task == "regression":
        # 回归任务处理
        result["prediction"] = pred_logits.item()  # 直接输出预测值

    return result

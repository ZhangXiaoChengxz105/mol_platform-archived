from GNN_data import smiles_to_graph
from GNN_model import GNN
import torch
import os
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
    path = os.path.join("models", "GNN_finetune", f"{name}_{target}.pth")
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
        result["confidence"] = max(pred_probs[0]).item()  # 最大概率值作为置信度
    elif model.task == "regression":
        # 回归任务处理
        result["prediction"] = pred_logits.item()  # 直接输出预测值

    return result

# test
smiles = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
# 分类任务测试
name = "BBBP"
target = "p_np"
print("\nClassification task test:")
class_result = gnn_predict(name, target, smiles)
print(f"SMILES: {class_result['smiles']}")
print(f"Name: {name}")
print(f"Target: {target}")
print(f"Task: {class_result['task']}")
print(f"Prediction/value: {class_result['prediction']}")
print(f"Label: {class_result['label']}")
print(f"Confidence: {class_result['confidence']}")

# 回归任务测试
name = "FreeSolv"
target = "expt"
print("\nRegression task test:")
class_result = gnn_predict(name, target, smiles)
print(f"SMILES: {class_result['smiles']}")
print(f"Name: {name}")
print(f"Target: {target}")
print(f"Task: {class_result['task']}")
print(f"Prediction/value: {class_result['prediction']}")
print(f"Label: {class_result['label']}")
print(f"Confidence: {class_result['confidence']}")




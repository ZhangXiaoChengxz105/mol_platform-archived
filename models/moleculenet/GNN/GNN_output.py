import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from GNN_data import smiles_to_graph
from GNN_model import GNN
from check_utils import get_datasets_measure_names, validate_datasets_measure_names

MODEL_LIST = ['GIN', 'GCN']
def predict(name, target, smiles_list, model_type = "GIN"):
    """
    简单的GNN预测接口
    Args:
        name: 属性名（如'BBBP'）
        path: 路径名 （如 models/finetune_gin/BBBP_best.pth）
        smiles: 分子SMILES字符串
    Returns:
        预测结果
    """
    if model_type not in MODEL_LIST:
        raise ValueError(f"模型类型错误: {model_type},\n可选模型类型: {MODEL_LIST}")
    
    validate_datasets_measure_names(name, target)
    
    path = os.path.join(models_root, "GNN_finetune", f"{model_type}",f"{name}_{target}.pth")
    print("Input path:", path)
    # 创建模型
    if os.path.exists(path):
        model = GNN(name, path, model_type)
        model.load_weights(path)
        print(".pth文件已加载，模型初始化成功")
    else:
        raise ValueError(f"模型文件不存在: {path}")
    
    graph_data = smiles_to_graph(smiles_list)
    predictions = model.predict(graph_data)
    # 预测
    results = []
    for i, smiles in enumerate(smiles_list):
        pred = predictions[i].item()
        
        result = {
            "data": smiles,
            "task": model.task,
            "prediction": pred,
            "label": None,
        }
        
        # 分类任务添加标签
        if model.task == "classification":
            result["label"] = 1 if pred > 0.5 else 0
        results.append(result)
    
    return results
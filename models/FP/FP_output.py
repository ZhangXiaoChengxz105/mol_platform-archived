import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from FP_data import smiles_to_fingerprint
from FP_model import FPModel
from check_utils import get_datasets_measure_names, validate_datasets_measure_names

def fp_predict(name, target, smiles_list):
    """
    FP预测主函数
    :param name: 数据集名称
    :param target: 预测目标
    :param smiles_list: SMILES列表
    """
    # 验证数据集和目标
    validate_datasets_measure_names(name, target)

    # 构建模型路径  
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "FP_finetune",
        f"{name}.pt"
    )
    print(f"加载模型: {model_path}")
    
    # 创建模型实例
    model = FPModel(name, model_path)
    
    # 加载模型权重
    model.load_weights()
    
    # 转换SMILES为指纹
    fp_tensor = smiles_to_fingerprint(smiles_list)
    
    # 预测
    predictions = model.predict(fp_tensor)
    
    # 格式化结果
    results = []
    for i, smiles in enumerate(smiles_list):
        measure_names = get_datasets_measure_names(name)
        index = measure_names.index(target)

        pred_value = predictions[i][index]
        
        result = {
            "smiles": smiles,
            "task": model.task,
            "prediction": pred_value,
            "label": None,
        }
        
        # 分类任务添加标签
        if model.task == "classification":
            result["label"] = 1 if pred_value > 0.5 else 0
        
        results.append(result)
    
    return results
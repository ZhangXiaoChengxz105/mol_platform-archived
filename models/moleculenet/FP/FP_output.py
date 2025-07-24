import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


import numpy as np
from FP_data import smiles_to_fingerprint
from FP_model import FP_NN
from models.check_utils import get_datasets_measure_names, validate_datasets_measure_names
from FP_model import FP_NN, FP_RF, FP_SVM, FP_XGB
MODEL_LIST = ['NN', 'RF', 'SVM', 'XGB']
MULTITASK_LIST = ['NN']
MODEL_MAP = {
    'NN': FP_NN,
    'RF': FP_RF,
    'SVM': FP_SVM,
    'XGB': FP_XGB
}

def predict(name, target, smiles_list, model_type = 'NN'):
    """
    FP预测主函数
    :param name: 数据集名称
    :param target: 预测目标
    :param smiles_list: SMILES列表
    """
    # 检验模型类型
    if model_type not in MODEL_LIST:
        raise ValueError(f"模型类型错误: {model_type},\n可选模型类型: {MODEL_LIST}")
    
    is_multitask = True if model_type in MULTITASK_LIST else False
    
    # 验证数据集和目标
    validate_datasets_measure_names(name, target)

    # 构建模型路径
    if model_type in MULTITASK_LIST:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "FP_finetune",
            model_type,
            f"{name}.pt"
        )
    else: 
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "FP_finetune",
            model_type,
            f"{name}_{target}.joblib"
        )
    print(f"加载模型: {model_path}")
    
    # 创建模型实例
    model = MODEL_MAP[model_type](name, model_path)
    
    # 加载模型权重
    model.load_weights()
    
    # 转换SMILES为指纹
    fp_tensor = smiles_to_fingerprint(smiles_list)
    
    # 预测
    predictions = model.predict(fp_tensor)
    
    # 格式化结果
    results = []
    for i, smiles in enumerate(smiles_list):
        if is_multitask:
            # 多任务模型 - 需要根据target找到对应的任务索引
            measure_names = get_datasets_measure_names(name)
            index = measure_names.index(target)
            pred_value = predictions[i][index]
        else:
            # 单任务模型 - 直接使用预测值
            pred_value = predictions[i]
        
        result = {
            "data": smiles,
            "task": model.task,
            "prediction": pred_value,
            "label": None,
        }
        
        # 分类任务添加标签
        if model.task == "classification":
            result["label"] = 1 if pred_value > 0.5 else 0
        
        results.append(result)
    
    return results
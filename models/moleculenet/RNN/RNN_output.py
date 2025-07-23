import os
import sys

# 添加上级目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from RNN.RNN_data import smiles_to_tensor
from RNN.RNN_model import RNNModel
from check_utils import validate_datasets_measure_names, get_datasets_measure_names

def predict(name, target, smiles_list, model_type = None):
    """
    RNN预测主函数
    :param name: 数据集名称（转换为小写）
    :param target: 预测目标
    :param smiles_list: SMILES列表
    """
    
    # 验证数据集和目标
    validate_datasets_measure_names(name, target)
    
    # 创建模型实例
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "RNN_finetune")
   
    model_filename = f"{name}.pt"
    model_path = os.path.join(model_dir, model_filename)
    model = RNNModel(name, model_path)
    
    # 加载模型权重（包括词汇表）
    model.load_weights()
    
    # 使用模型中的词汇表转换SMILES为张量
    sequences, lengths = smiles_to_tensor(
        smiles_list, 
        model.char_to_index
    )
    
    # 预测
    predictions = model.predict(sequences, lengths)
    
    # 格式化结果
    results = []
    measure_names = get_datasets_measure_names(name)
    target_idx = measure_names.index(target)
    
    for i, smiles in enumerate(smiles_list):
        pred_value = predictions[i][target_idx].item()
        
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
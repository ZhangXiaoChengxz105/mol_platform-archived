import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from SEQ_data import smiles_to_tokens
from SEQ_model import SEQ as SEQ
from models.check_utils import get_datasets_measure_names, validate_datasets_measure_names

def predict(name, target, smiles_list, model_type = None):
    validate_datasets_measure_names(name, target)
    # 构建绝对路径（假设models在项目根目录下）
    if name not in ["Tox21", "ClinTox","MUV","SIDER"]:
        path = os.path.join(models_root, "SEQ_finetune", f"{name}_{target}.pth")
    else:
        path = os.path.join(models_root, "SEQ_finetune", f"{name}.pth")
    print("\nInput path:", path)
    if os.path.exists(path):
        model = SEQ(name, path)
        model.load_weights()
        print(".pth文件已加载，模型初始化成功")
    else:
        raise ValueError(f"模型文件不存在: {path}")
    # 参数文件转换：
    # torch.save(model.model.state_dict(), path)
    # print(f"模型权重已保存为 {path}")

    # if os.path.exists(path_ckpt)&os.path.exists(path):
    #     os.remove(path_ckpt)
    #     print(f"已删除重复.ckpt 文件: {path_ckpt}")

    # 3. 预测结果
    # 数据转换
    
    token_data = smiles_to_tokens(smiles_list)
    predictions = model.predict(token_data)
    results = []
    for i, smiles in enumerate(smiles_list):
        pred_value = predictions[i]
        if name in ["Tox21", "ClinTox","MUV","SIDER"]:
            measure_names = get_datasets_measure_names(name)
            index = measure_names.index(target)
            pred_value = pred_value[index]
        pred_value = pred_value.item()
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

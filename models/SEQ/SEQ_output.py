import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SEQ_data import smiles_to_tokens
from SEQ_model import SEQ
from check_utils import get_datasets_measure_names, validate_datasets_measure_names

from time import time
def seq_predict(name, target, smiles):
    validate_datasets_measure_names(name, target)
    
    if name not in ["Tox21", "ClinTox","MUV","SIDER"]:
        path_prefix = os.path.join("models", "SEQ_finetune", f"{name}_{target}")
    else:
        path_prefix = os.path.join("models", "SEQ_finetune", f"{name}")

    path_pth = os.path.join(path_prefix + ".pth")
    path_cktp = os.path.join(path_prefix + ".ckpt")
    if os.path.exists(path_pth):
        model = SEQ(name, path_pth)
        model.load_weights()
        print(".pth文件已加载，模型初始化成功")
    elif os.path.exists(path_cktp):
        model = SEQ(name, path_cktp)
        model.load_weights()
        print(".ckpt文件已加载，模型初始化成功")
    else:
        raise ValueError(f"模型文件不存在: {path_pth} or {path_cktp}")
    
    # 参数文件转换：
    # torch.save(model.model.state_dict(), path_pth)
    # print(f"模型权重已保存为 {path_pth}")

    # if os.path.exists(path_cktp)&os.path.exists(path_pth):
    #     os.remove(path_cktp)
    #     print(f"已删除重复.ckpt 文件: {path_cktp}")

    # 3. 预测结果
    # 数据转换
    token_data = smiles_to_tokens(smiles)
    print("\n\nsmiles转换后的tokens_emb: [input_ids, attention_mask]")
    print(token_data)
    
    result = {
        "smiles": smiles,
        "task": model.task,
        "prediction": None,
        "label": None,
    }
    if name not in ["Tox21", "ClinTox","MUV","SIDER"]:
        pred = model.predict(token_data)
    else:
        pred_list = model.predict(token_data)
        print(f"\n\n\npred_list:{pred_list}\n\n\n")
        measure_names = get_datasets_measure_names(name)
        i = measure_names.index(target)
        pred = pred_list[0][i]


    if model.task == "regression":
        # 回归任务或多属性分类任务处理
        result["prediction"] = pred.item()  # 直接输出预测值
    else:
        pred_probs = torch.softmax(pred, dim=1)[0][1] if model.task == "classification" else pred
        result["prediction"] = pred_probs.item()  # 正类概率
        result["label"] = 1 if result["prediction"] > 0.5 else 0
        
    
    # 4. 输出结果 (p-np概率值)
    return result

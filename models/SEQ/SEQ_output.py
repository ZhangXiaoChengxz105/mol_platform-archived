import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SEQ_data import smiles_to_tokens
from SEQ_model import SEQ

def seq_predict(name, target, smiles):
    path_prefix = os.path.join("models", "SEQ_finetune", f"{name}_{target}")
    path_pth = os.path.join(path_prefix+".pth")
    path_cktp = os.path.join(path_prefix+".ckpt")
    if os.path.exists(path_pth):
        model = SEQ(name, path_pth)
        model.load_weights()
        print(".pth文件已加载")
    elif os.path.exists(path_cktp):
        model = SEQ(name, path_cktp)
        model.load_weights()
        print(".ckpt文件已加载")
        save_path = os.path.join("models", "SEQ_finetune", f"{name}_{target}.pth")
        torch.save(model.model.state_dict(), save_path)
        print(f"模型权重已保存为 {save_path}")

    else:
        raise ValueError(f"模型文件不存在: {path_pth} or {path_cktp}")
    
    if os.path.exists(path_cktp)&os.path.exists(path_pth):
        os.remove(path_cktp)
        print(f"已删除重复.ckpt 文件: {path_cktp}")
    # 3. 预测结果
    # 数据转换
    token_data = smiles_to_tokens(smiles)
    print("\n\n\nsmiles_emv:")
    print(token_data)
    print(token_data[0].shape)
    print("\n\n\n")
    
    pred_logits = model.predict(token_data)
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
    
    # 4. 输出结果 (p-np概率值)
    return result


# 示例使用

if __name__ == "__main__":
    test_smile = "C1=CC=CC=C1"  # 苯环
    name = "HIV"
    target = "HIV_active"
    result = seq_predict(name,target,test_smile)
    print(result)

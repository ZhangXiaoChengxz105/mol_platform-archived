# This model prediction workflow is based on the following paper:
Large-scale chemical language 
representations capture molecular structure 
and properties
https://www.nature.com/articles/s42256-022-00580-7

# Contributor and contribution:
Yanzhen, Chen
Zhejiang University
contribution includes: Biult the workflow, finetuned origin model's parameter and simplified origin paper's models to "SEQ" for prediction

# **环境配置**
    python=3.11.8
    
    pip install torch torch-geometric rdkit==2024.3.5 numpy scikit-learn==1.7.0 transformers pandas xgboost
	
# **模块结构**
    moleculenet/
    ├── SEQ/                        # 模型核心文件
    │   ├── ginet_finetune.py       # 模型训练入口
    │   ├── SEQ_data.py             # 数据预处理
    │   ├── SEQ_model.py            # 模型核心实现
    │   ├── SEQ_output.py           # 预测接口
    │   └── SEQ_test.py
    ├── SEQ_finetune/               # 预训练参数
    │   ├── BACE_Class.pth          # 各任务参数文件
    │   └── ...

## **核心文件**：
### SEQ_data.py
#### convert smiles to sequence:
    canon_smi = canonicalize_smiles(smiles)
    if not canon_smi:
        raise ValueError("SMILES标准化失败")
    
    # 直接使用原始tokenizer（引用自）
    tokenizer = TOKENIZER
    tokens = tokenizer.batch_encode_plus(
        [smiles] if isinstance(smiles, str) else smiles,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    data = [input_ids, attention_mask]
    data_list.append(data)
### SEQ_model.py
#### 4 kinds of models:
	SEQ_NN    # Neuron Network
	SEQ_RF    # Random Forest
	SEQ_SVM   # Support Vector Machine
	SEQ_XGB   # XGBoost
	
	model input: fingerprints list (corresponding to smiles)

### SEQ_output.py
#### formalized model output:
	results = []
	result = {
            "data": smiles,
            "task": model.task,
            "prediction": pred_value,
            "label": None,
    }
	return results

# **快速开始 （代码复用）**
## 1. 数据准备
按照模块结构，解压SEQ_finetune至moleculenet文件夹中

### 推荐使用如下命令添加路径：
from models.moleculenet.SEQ.SEQ_output import predict  # 根据自己的路径调整
dir(dir(dir...(你的文件位置)))直到找到models上一级目录（根目录）
ex: sys.path.append(os.path.dirname(os.path.abspath(__file__)))

## 2. 模型工作流使用
### 样本预测示例
    smile1 = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    smile2 = "CN(C)CCCN1c2ccccc2Sc3ccc(cc13)C(F)(F)F"
    smiles_list = [smile1,smile2]

    name = "BBBP"
    target = "p_np"

    results = predict(name, target, smiles_list, model_type = 'GIN')
### 输入结构
    smiles_list = [
        "smile1",
        "smile2",
        ...
    ]
### 输出结构
    results = [
        {
            "data": smile1,
            "task": ,
            "prediction": ,
            "label":
        },
        {
            "data": smile2,
            "task": ,
            "prediction": ,
            "label":
        }
    ]
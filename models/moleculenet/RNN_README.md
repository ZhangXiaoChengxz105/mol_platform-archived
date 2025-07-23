# This model prediction workflow is partially based on the following paper:
A systematic study of key elements underlying molecular property prediction
https://www.nature.com/articles/s41467-023-41948-6
# Contributor and contribution:
Yanzhen, Chen
Zhejiang University
contribution includes: Biult the workflow, trained model's parameter and simplified it for prediction usage

# **模块结构**
	moleculnet/
	├── RNN/                        # 模型核心文件
	│   ├── ginet_finetune.py      # 模型训练入口
	│   ├── RNN_data.py             # 数据预处理
	│   ├── RNN_model.py            # 模型核心实现
	│   ├── RNN_output.py           # 预测接口
	│   └── RNN_test.py

	├── RNN_finetune/         		# 预训练参数
	│   ├── BACE.pt             # 各任务参数文件
	│   └── ...

## **核心文件**：
### RNN_data.py
#### convert smiles to fingerprints(mixed, dim = 1489):
    sequences, lengths = [], []
    for smile in smiles_list:
        # 过滤无效字符
        seq = [char_to_index.get(c, 0) for c in smile][:max_length]
        sequences.append(seq)
        lengths.append(len(seq))
        
    # 填充序列
    max_len = max(lengths) if lengths else 0
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        
    return torch.tensor(padded), torch.tensor(lengths)

### RNN_model.py
    RNN     # wrapper of simplified model from origin paper
### RNN_output.py
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
按照模块结构，解压RNN_finetune至moleculenet文件夹中

### 推荐使用如下命令添加路径：
from models.moleculent.RNN.RNN_output import predict  # 根据自己的路径调整
dir(dir(dir...(你的文件位置)))直到找到models上一级目录（根目录）
ex: sys.path.append(os.path.dirname(os.path.abspath(__file__)))

## 2. 模型工作流使用
### 样本预测示例
	smile1 = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
	smile2 = "CN(C)CCCN1c2ccccc2Sc3ccc(cc13)C(F)(F)F"
	smiles_list = [smile1,smile2]

	name = "BBBP"
	target = "p_np"
	results = predict(name, target, smiles_list, model_type = 'NN')

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
### 模型对应关系
model_type 类型：
	MODEL_MAP = {
		'NN': RNN_NN,    # Neuron Network
		'RF': RNN_RF,    # Random Forest
		'SVM': RNN_SVM,  # Support Vector Machine
		'XGB': RNN_XGB   # XGBoost
	}
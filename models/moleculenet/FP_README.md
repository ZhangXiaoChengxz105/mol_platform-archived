# This model prediction workflow is partially based on the following paper:
One of the models, "FP_NN" is based on the following paper:
FP-GNN: a versatile deep learning architecture for enhanced molecular property prediction
https://academic.oup.com/bib/article/23/6/bbac408/6702671
# Contributor and contribution:
Yanzhen, Chen
Zhejiang University
contribution includes: Biult the workflow, finetuned origin model's parameter and simplified origin paper's models to "FP_GN" for prediction, designed and trained the models "FP_RF", "FP_SVM", "FP_XGB" from tradition ML method(RF, SVM, XGB)

# **环境配置**
    python=3.11.8
    
    pip install torch torch-geometric rdkit==2024.3.5 numpy scikit-learn==1.7.0 transformers pandas xgboost
	
# **模块结构**
	moleculnet/
	├── FP/                        # 模型核心文件
	│   ├── ginet_finetune.py      # 模型训练入口
	│   ├── FP_data.py             # 数据预处理
	│   ├── FP_model.py            # 模型核心实现
	│   ├── FP_output.py           # 预测接口
	│   └── FP_test.py

	├── FP_finetune/         		# 预训练参数
	│   ├── NN/                    	# 模型参数文件夹
	│   │   ├── BACE.pt             # 各任务参数文件
	│   │   └── ...
	│   ├── RF/
	│   │   ├── BACE_Class.joblib   # 各任务参数文件
		...

## **核心文件**：
### FP_data.py
#### convert smiles to fingerprints(mixed, dim = 1489):
	mol = Chem.MolFromSmiles(smile)
	fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))
	fp_phaErGfp = list(AllChem.GetErGFingerprint(
		mol, fuzzIncrement=0.3, maxPath=21, minPath=1
	))
	fp_pubcfp = list(GetPubChemFPs(mol))
	fp = fp_maccs + fp_phaErGfp + fp_pubcfp # dim = 1489

### FP_model.py
#### 4 kinds of models:
	FP_NN    # Neuron Network
	FP_RF    # Random Forest
	FP_SVM   # Support Vector Machine
	FP_XGB   # XGBoost
	
	model input: fingerprints list (corresponding to smiles)

### FP_output.py
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
按照模块结构，解压FP_finetune至moleculenet文件夹中

### 推荐使用如下命令添加路径：
from models.moleculent.FP.FP_output import predict  # 根据自己的路径调整
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
		'NN': FP_NN,    # Neuron Network
		'RF': FP_RF,    # Random Forest
		'SVM': FP_SVM,  # Support Vector Machine
		'XGB': FP_XGB   # XGBoost
	}
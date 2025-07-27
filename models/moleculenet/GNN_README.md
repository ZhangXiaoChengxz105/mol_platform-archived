# This model prediction workflow is based on the following paper:
Molecular contrastive learning of representations 
via graph neural networks
https://www.nature.com/articles/s42256-022-00447-x

# Contributor and contribution:
Yanzhen, Chen
Zhejiang University
contribution includes: Biult the workflow, finetuned origin model's parameter and simplified origin paper's models to "GNN" for prediction

# **环境配置**
    python=3.11.8
    
    pip install torch torch-geometric rdkit==2024.3.5 numpy scikit-learn==1.7.0 transformers pandas xgboost
可根据GNN_requirements.txt(模型依赖), 使用env_utils.py，快速创建独立模型环境，安装依赖

    python env_utils.py create -a models/moleculenet/GNN_requirements.txt -e your_env_name -p 3.11.8
环境创建后请在使用平台时指定使用模型工作流对应的环境

# **模块结构**
    moleculenet/
    ├── GNN/                        # 模型核心文件
    │   ├── ginet_finetune.py       # 模型训练入口
    │   ├── GNN_data.py             # 数据预处理
    │   ├── GNN_model.py            # 模型核心实现
    │   ├── GNN_output.py           # 预测接口
    │   └── GNN_test.py

	├── GNN_finetune/         		# 预训练参数
	│   ├── GIN/                # 模型参数文件夹
	│   │   ├── BACE_Class.pth      # 各任务参数文件
	│   │   └── ...
	│   ├── GCN/
	│   │   ├── BACE_Class.pth      # 各任务参数文件
		...
## **核心文件**：
### GNN_data.py
#### convert smiles to graph data:   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 添加氢原子（保持与训练一致）
    mol = Chem.AddHs(mol)

    # 原子特征处理
    atom_features = []
    for atom in mol.GetAtoms():
        # 原子类型映射
        atomic_num = atom.GetAtomicNum()
        try:
            type_idx = ATOM_LIST.index(atomic_num)
        except ValueError:
            type_idx = len(ATOM_LIST) - 1  # 未知原子映射到最后一项
        
        # 手性特征映射
        try:
            chirality_idx = CHIRALITY_LIST.index(atom.GetChiralTag())
        except ValueError:
            chirality_idx = 0  # 未知手性映射到CHI_UNSPECIFIED
        
        atom_features.append([type_idx, chirality_idx])

    # 边特征处理
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # 键类型映射
        try:
            bond_type_idx = BOND_LIST.index(bond.GetBondType())
        except ValueError:
            bond_type_idx = 0
        
        # 键方向映射
        try:
            bond_dir_idx = BONDDIR_LIST.index(bond.GetBondDir())
        except ValueError:
            bond_dir_idx = 0
        
        # 双向添加（保持图的无向性）
        row += [start, end]
        col += [end, start]
        edge_feat.append([bond_type_idx, bond_dir_idx])
        edge_feat.append([bond_type_idx, bond_dir_idx])

    # 创建数据对象
    data =  Data(
        x=torch.tensor(atom_features, dtype=torch.long),
        edge_index=torch.tensor([row, col], dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.long),
        # 预测不需要真实标签，但为了兼容性保留y属性
        y=None  # 虚拟标签
    )

### GNN_model.py
#### 2 kinds of Net in models:
	GNN_GNN    # Graph Neural Network
    1. GIN    
    2. GCN
	model input: graph data list (corresponding to smiles)

### GNN_output.py
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
按照模块结构，解压GNN_finetune至moleculenet文件夹中

### 推荐使用如下命令添加路径：
from models.moleculenet.GNN.GNN_output import predict  # 根据自己的路径调整

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
### 模型对应关系
model_type 类型：
	MODEL_LIST = ['GIN', 'GCN']



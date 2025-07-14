# **模块结构**
    models/
    ├── GNN/                        # 模型核心文件
    │   ├── ginet_finetune.py       # 模型训练入口
    │   ├── GNN_data.py             # 数据预处理
    │   ├── GNN_model.py            # 模型核心实现
    │   ├── GNN_output.py           # 预测接口
    │   └── GNN_test.py

	├── FP_finetune/         		# 预训练参数
	│   ├── GIN/                # 模型参数文件夹
	│   │   ├── BACE_Class.pth      # 各任务参数文件
	│   │   └── ...
	│   ├── GCN/
	│   │   ├── BACE_Class.pth      # 各任务参数文件
		...

# **快速开始**
## 1. 数据准备
按照模块结构，解压GNN_finetune至models文件夹中

### 推荐使用如下命令添加路径：
from models.GNN.GNN_output import GNN_predict  # 根据自己的路径调整

dir(dir(dir...(你的文件位置)))直到找到models上一级目录（根目录）

ex: sys.path.append(os.path.dirname(os.path.abspath(__file__)))

## 2. 初始化模型（自动加载对应任务的预训练参数）
    GNN = GNN(task_name="BACE", modeltype = “GIN”)  # 支持BBBP/HIV等GNN_finetune目录下GIN或GCN模型的任务

## 3. 样本预测
    smile1 = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    smile2 = "CN(C)CCCN1c2ccccc2Sc3ccc(cc13)C(F)(F)F"
    smiles_list = [smile1,smile2]

    name = "BBBP"
    target = "p_np"

    results = GNN_predict(name, target, smiles_list, model_type = 'GIN')
### 输出结构
    results = [
        {
            "smiles": smile1,
            "task": ,
            "prediction": ,
            "label":
        },
        {
            "smiles": smile2,
            "task": ,
            "prediction": ,
            "label":
        }
    ]
### 模型对应关系
model_type 类型：
	MODEL_LIST = ['GIN', 'GCN']



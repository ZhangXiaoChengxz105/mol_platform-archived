1. 模块结构
models/
├── GNN/
│   ├── ginet_finetune.py    # 模型训练入口
│   ├── GNN_data.py          # 数据预处理
│   ├── GNN_model.py         # 模型核心实现
│   ├── GNN_output.py        # 预测接口
|   ├── GNN_test.py          # 测试接口
├── GNN_finetune/            # 预训练参数
│   ├── BACE_Class.pth       # 各任务参数文件
│   └── ...

2. 快速开始
from models.GNN.GNN_output import gnn_predict

# 初始化模型（自动加载对应任务的预训练参数）
gnn = GNN(task_name="BACE")  # 支持BBBP/HIV等GNN_finetune目录下的任务

# 单样本预测
smiles = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
name = "BBBP"
target = "p_np"
result = gnn_predict(name, target, smiles)
'''
result = {
        "smiles": ,
        "task": ,
        "prediction": ,
        "label": ,
        "confidence": 
    }
'''




import torch
from torch_geometric.data import Batch

from ginet_finetune import GINet

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import base_model
'''
class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.task = task(name)
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass
'''


class GNN(base_model):
    def __init__(self, name, path = None):
        super().__init__(name, path)      
        # 初始化模型结构
        self.model = GINet(task=self.task)
    
    def load_weights(self, path=None):
        """实现微调模型加载逻辑"""
        if path is None:
            if self.path is None:
                print("模型未初始化或添加路径，无法加载")
            else: 
                path = self.path
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
    
    def predict(self, data):
        self.model.eval()
        if not hasattr(data, 'batch') or data.batch is None:
            batch_data = Batch.from_data_list([data])
        with torch.no_grad():
            _, pred = self.model(batch_data) # torch_geometric.data.Data
        return pred
'''
# 使用示例
bbbp_model = GraphModel("BBBP", "models/")  # 自动设置task='classification'
qm9_model = GraphModel("QM9", "models/")    # 自动设置task='regression'

# 直接预测
result = bbbp_model.predict(molecule_graph)
'''

'''from torch_geometric.data import Data

data = Data(
    x=x,  # 节点特征矩阵，形状 [num_nodes, num_node_features]
    edge_index=edge_index,  # 边索引，形状 [2, num_edges]
    edge_attr=edge_attr,  # 边特征，形状 [num_edges, num_edge_features]
    y=label  # 可选，标签
)'''

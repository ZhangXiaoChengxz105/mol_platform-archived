import torch
from ginet_finetune import GINet
from base_model import base_model

class GNN(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        self.name = name

        # 根据属性自动决定task
        if name in ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE', 'SIDER', 'MUV', 'PCBA']:
            self.task = 'classification'
        else: self.task = 'regression'
        
        # 初始化模型结构
        self.model = GINet(task=self.task)
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
    
    def predict(self, data):
        self.model.eval()
        from torch_geometric.data import Batch
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

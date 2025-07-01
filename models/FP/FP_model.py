import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from base_model import base_model
from check_utils import get_datasets_measure_numbers

class FPModel(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        self.fp_dim = 1489  # PubChem混合指纹维度
        print(self.name)
        # 根据任务类型确定输出维度
        if self.task == "classification" and self.name not in ["Tox21", "ClinTox", "MUV", "SIDER"]:
            output_dim = 1
        else:
            output_dim = get_datasets_measure_numbers(name)
            print(f"multitask: {output_dim} tasks")

            
        # 创建模型实例（完全匹配权重文件的结构）
        self.model = self.Net(
            fp_dim=self.fp_dim,
            output_dim=output_dim
        )
        
        # 分类任务需要sigmoid
        if self.task == "classification":
            self.sigmoid = nn.Sigmoid()

    def load_weights(self, path=None):
        """加载模型权重"""
        load_path = path if path else self.path
        if not load_path.endswith(".pt"):
            raise ValueError("模型路径必须以.pt结尾")
        
        try:
            checkpoint = torch.load(load_path, weights_only = False)
            # 严格匹配权重
            state_dict = checkpoint['state_dict']
            self.model.data_scaler = checkpoint['data_scaler']
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"FP模型权重已从 {load_path} 加载")
        except Exception as e:
            # 打印详细错误信息帮助调试
            print(f"详细错误信息: {str(e)}")
            raise RuntimeError(f"权重加载失败: {str(e)}")

    def predict(self, fp_tensor):
        """预测接口"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(fp_tensor)
            
            # 分类任务应用sigmoid
            if self.task == "classification":
                predictions = self.sigmoid(predictions)
            return predictions

    # 完全匹配权重文件的结构
    class Net(nn.Module):
        def __init__(self, fp_dim, output_dim, dropout=0.2):
            super().__init__()
            # 指纹处理网络 (FPN)
            self.fpn = nn.Sequential(
                nn.Linear(fp_dim, 512),  # 完全匹配权重文件维度
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 300)     # 完全匹配权重文件维度
            )
            
            # 前馈网络 (FFN)
            self.ffn = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(300, 300),    # 完全匹配权重文件维度
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(300, output_dim)  # 输出层维度根据任务调整
            )
            self.data_scaler = None
        
        def forward(self, fp_tensor):
            """前向传播"""
            output = self.fpn(fp_tensor)
            output = self.ffn(output)
            print(self.data_scaler)
            if self.data_scaler:
                output = output*self.data_scaler[1] + self.data_scaler[0]
            return output
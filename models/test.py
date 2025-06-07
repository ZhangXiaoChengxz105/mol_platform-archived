import yaml
import numpy as np
from models import FingerprintModel, SequenceModel, GraphModel

# 示例数据
smiles_list = ["CCO", "CCN", "CC(=O)O"]
labels = np.array([0, 1, 0])

# 初始化模型
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

fp_model = FingerprintModel("fingerprint", "models/fingerprint")
seq_model = SequenceModel("sequence", "models/sequence")
graph_model = GraphModel("graph", "models/graph")

# 训练（以指纹模型为例）
X_fp = [fp_model.featurize(smi) for smi in smiles_list]
fp_model.train(X_fp, labels)

# 预测
preds_fp = fp_model.predict(smiles_list)
preds_seq = seq_model.predict(smiles_list)
preds_graph = graph_model.predict(smiles_list)

print("Fingerprint predictions:", preds_fp)
print("Sequence predictions:", preds_seq)
print("Graph predictions:", preds_graph)

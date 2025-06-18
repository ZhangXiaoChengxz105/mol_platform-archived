import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN_output import gnn_predict


def test_prediction(name, target, smiles):
    class_result = gnn_predict(name, target, smiles)
    print(f"SMILES: {class_result['smiles']}")
    print(f"Name: {name}")
    print(f"Target: {target}")
    print(f"Task: {class_result['task']}")
    print(f"Prediction/value: {class_result['prediction']}")
    print(f"Label: {class_result['label']}")
    print(f"Confidence: {class_result['confidence']}")
    return class_result


if __name__ == "__main__":
    # test
    smiles = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    # 分类任务测试
    name = "BBBP"
    target = "p_np"
    print("\nClassification task test:")
    test_prediction(name, target, smiles)

    # 回归任务测试
    name = "FreeSolv"
    target = "expt"
    print("\nRegression task test:")
    test_prediction(name, target, smiles)
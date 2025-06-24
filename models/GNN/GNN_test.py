import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN_output import gnn_predict


def test_prediction(name, target, smiles_list):
    results = gnn_predict(name, target, smiles_list)
    print(f"\n{name}_{target}_results:")
    for i in range(len(smiles_list)):
        result = results[i]
        print(f"\nSMILES: {result['smiles']}")
        print(f"Name: {name}")
        print(f"Target: {target}")
        print(f"Task: {result['task']}")
        print(f"Prediction/value: {result['prediction']}")
        print(f"Label: {result['label']}")
    return results


if __name__ == "__main__":
    # test
    smile1 = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    smile2 = "CN(C)CCCN1c2ccccc2Sc3ccc(cc13)C(F)(F)F"
    smiles_list = [smile1,smile2]
    # 分类任务测试
    name = "BBBP"
    target = "p_np"
    print("\nClassification task test:")
    test_prediction(name, target, smiles_list)

    # 回归任务测试
    name = "FreeSolv"
    target = "expt"
    print("\nRegression task test:")
    test_prediction(name, target, smiles_list)
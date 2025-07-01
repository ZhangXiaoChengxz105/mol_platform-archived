import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN_output import gnn_predict


def test_prediction(name, target, smiles_list):
    print(f"\n测试 {name} 数据集, 目标: {target}")
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
    smile1 = "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
    smile2 = "CN(C)C(=O)c1ccc(cc1)OC"
    smile3 = "[H]C([H])([H])C([H])([H])[H]"
    smiles_list = [smile1,smile2]
    # 分类任务测试
    # name = "BBBP"
    # target = "p_np"
    # print("\nClassification task test:")
    # test_prediction(name, target, smiles_list)

    # 回归任务测试
    # name = "FreeSolv"
    # target = "expt"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2]) # true value: -11.01

    
    # name = "Lipo"
    # target = "exp"
    # print("\nRegression task test:")
    # test_prediction(name, target, smiles_list) # true value: 3.54

    name = "qm7"
    target = "u0_atom"
    print("\nRegression task test:")
    test_prediction(name, target, [smile2, smile3]) # true value: -712.42

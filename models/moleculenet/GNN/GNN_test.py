import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GNN_output import predict


def test_prediction(name, target, smiles_list, model_type = "GCN"):
    print(f"\n测试 {name} 数据集, 目标: {target}, 模型类型: {model_type}")
    results = predict(name, target, smiles_list, model_type)
    print(f"\n{name}_{target}_results:")
    for i in range(len(smiles_list)):
        result = results[i]
        print(f"\nSMILES: {result['data']}")
        print(f"Name: {name}")
        print(f"Target: {target}")
        print(f"Task: {result['task']}")
        print(f"Prediction/value: {result['prediction']}")
        print(f"Label: {result['label']}")
    return results


if __name__ == "__main__":
    # test
    smile1 = "C1CCN(CC1)Cc1cccc(c1)OCCCNC(=O)C"
    smile2 = "CN(C)C(=O)c1ccc(cc1)OC"
    smile3 = "[H]C([H])([H])C([H])([H])[H]"
    smiles_list = [smile1,smile2]

    model_type = "GCN"
    #分类任务测试
    name = "BBBP"
    target = "p_np"
    print("\nClassification task test:")
    test_prediction(name, target, smiles_list, model_type = "GCN") # true value: 1 for smile1

    # 回归任务测试
    name = "FreeSolv"
    target = "expt"
    print("\nRegression task test:")
    test_prediction(name, target, [smile2], "GCN") # true value: -11.01

    
    # name = "Lipo"
    # target = "exp"
    # print("\nRegression task test:")
    # test_prediction(name, target, smiles_list) # true value: 3.54

    # name = "qm7"
    # target = "u0_atom"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3]) # true value: -712.42
    
    # name = "ESOL"
    # target = 'measured log solubility in mols per litre'
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3]) # true value: -4.00

    # name = "qm8"
    # target = "f1-CAM"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3])

    name = "SIDER"
    target = "Blood and lymphatic system disorders"
    print("\nRegression task test:")
    test_prediction(name, target, [smile2, smile3], "GCN")
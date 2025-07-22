import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SEQ_output import seq_predict


def test_prediction(name, target, smiles_list):
    print(f"\n测试 {name} 数据集, 目标: {target}")
    results = seq_predict(name, target, smiles_list)
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
    smile1 = "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
    smile2 = "CN(C)C(=O)c1ccc(cc1)OC"
    smile3 = "[H]C([H])([H])C([H])([H])[H]"
    smiles_list = [smile1,smile2]
    # # 分类任务测试
    name = "BBBP"
    target = "p_np"
    print("\n\nClassification task test:")
    test_prediction(name, target, smiles_list)
    # # 分类多任务测试
    # name = "ClinTox"
    # print("\n\nClassification multitask test:")
    # target = "FDA_APPROVED"
    # test_prediction(name, target, smiles_list)
    
    # target = "CT_TOX"
    # test_prediction(name, target, smiles_list)
    # # 回归任务测试
    # name = "FreeSolv"
    # target = "expt"
    # print("\n\nRegression task test:")
    # test_prediction(name, target, [smile2])

    # name = "qm7"
    # target = "u0_atom"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile3]) # -712.42

    # name = "qm8"
    # target = "f1-CAM"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3])
    
    # name = "SIDER"
    # target = "Hepatobiliary disorders"
    # print("\nClassification task test:")
    # test_prediction(name, target, [smile2, smile3])

    name = "Tox21"
    target = "NR-AR"
    print("\n\nClassification task test:")
    test_prediction(name, target, smiles_list)
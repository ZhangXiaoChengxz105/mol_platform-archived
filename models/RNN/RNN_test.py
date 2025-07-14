import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RNN.RNN_output import rnn_predict

def test_prediction(name, target, smiles_list):
    print(f"\n测试 {name} 数据集, 目标: {target}")
    results = rnn_predict(name, target, smiles_list)
    print(f"\n{name}_{target}_results:")
    for i in range(len(smiles_list)):
        result = results[i]
        print(f"\nSMILES: {result['smiles']}")
        print(f"数据集: {name}")
        print(f"目标属性: {target}")
        print(f"任务类型: {result['task']}")
        print(f"预测值: {result['prediction']}")
        print(f"分类标签: {result['label']}")
    return results

if __name__ == "__main__":
    # 测试分子
    smile1 = "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"  # 有效分子
    smile2 = "CN(C)C(=O)c1ccc(cc1)OC"                   # 有效分子
    smile3 = "[H]C([H])([H])C([H])([H])[H]"             # 无效分子（乙烷）
    smiles_list = [smile1, smile2, smile3]
    
    # 分类任务测试 - BBBP
    name = "BBBP"
    target = "p_np"
    print("\n\n分类任务测试 (BBBP - p_np):")
    test_prediction(name, target, smiles_list)
    
    # 分类多任务测试 - ClinTox
    name = "ClinTox"
    print("\n\n分类多任务测试 (ClinTox):")
    target = "FDA_APPROVED"
    test_prediction(name, target, smiles_list)
    target = "CT_TOX"
    test_prediction(name, target, smiles_list)
    
    # # 回归任务测试 - FreeSolv
    # name = "FreeSolv"
    # target = "expt"
    # print("\n\n回归任务测试 (FreeSolv - expt):")
    # test_prediction(name, target, smiles_list)
    
    # name = "qm7"
    # target = "u0_atom"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3]) # true value: -712.42
    
    # # 回归任务测试 - qm8
    # name = "qm8"
    # target = "f1-CAM"
    # print("\n\n回归任务测试 (qm8 - f1-CAM):")
    # test_prediction(name, target, smiles_list)
    
    # # 多目标分类测试 - Tox21
    # name = "Tox21"
    # target = "NR-AR"
    # print("\n\n多目标分类测试 (Tox21 - NR-AR):")
    # test_prediction(name, target, smiles_list)
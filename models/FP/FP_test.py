import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FP_output import fp_predict

def test_prediction(name, target, smiles_list, model_type = 'NN'):
    """测试FP预测"""
    print(f"\n测试 {name} 数据集, 目标: {target}, 模型类型: {model_type}")
    results = fp_predict(name, target, smiles_list, model_type = model_type)
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
    # 测试数据
    smile1 = "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
    smile2 = "CN(C)C(=O)c1ccc(cc1)OC"
    smile3 = "[H]C([H])([H])C([H])([H])[H]"
    smile4 = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    smiles_list = [smile1,smile2]
    
    # 分类任务测试
    name = "BBBP"
    print(f"\n分类任务测试 ({name}):")
    test_prediction(name, "p_np", [smile4], model_type='RF')
    
    test_prediction(name, "p_np", [smile4], model_type='SVM')
    
    test_prediction(name, "p_np", [smile4], model_type='XGB')
    
    # # 多任务分类测试
    # name = "ClinTox"
    # print(f"\n分类任务测试 ({name}):")
    # test_prediction(name, "FDA_APPROVED", smiles_list)
    # test_prediction(name, "CT_TOX", smiles_list)
    
    # # 回归任务测试
    # name = "FreeSolv"
    # target = "expt"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2]) # true value: -11.01

    
    # name = "Lipo"
    # target = "exp"
    # print("\nRegression task test:")
    # test_prediction(name, target, smiles_list) # true value: 3.54

    # name = "qm7"
    # target = "u0_atom"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile3]) # true value: -712.42

    # name = "qm8"
    # target = "f1-CAM"
    # print("\nRegression task test:")
    # test_prediction(name, target, [smile2, smile3]) # true value: -712.42
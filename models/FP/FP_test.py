import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FP_output import fp_predict

def test_fp_prediction(name, target, smiles_list):
    """测试FP预测"""
    print(f"\n测试 {name} 数据集, 目标: {target}")
    results = fp_predict(name, target, smiles_list)
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
    smile1 = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    smile2 = "CN(C)CCCN1c2ccccc2Sc3ccc(cc13)C(F)(F)F"
    smiles_list = [smile1,smile2]
    
    # 分类任务测试
    # name = "BBBP"
    # print(f"\n分类任务测试 ({name}):")
    # test_fp_prediction(name, "p_np", smiles_list)
    
    # # 多任务分类测试
    # name = "ClinTox"
    # print(f"\n分类任务测试 ({name}):")
    # test_fp_prediction(name, "FDA_APPROVED", smiles_list)
    # test_fp_prediction(name, "CT_TOX", smiles_list)
    
    # 回归任务测试
    name = "FreeSolv"
    print(f"\n回归任务测试 ({name}):")
    test_fp_prediction(name, "expt", smiles_list)
    
    name = "qm7"
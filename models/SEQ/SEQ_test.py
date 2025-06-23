import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SEQ_output import seq_predict


def test_prediction(name, target, smiles):
    result = seq_predict(name, target, smiles)
    print(f"SMILES: {result['smiles']}")
    print(f"Name: {name}")
    print(f"Target: {target}")
    print(f"Task: {result['task']}")
    print(f"Prediction/value: {result['prediction']}")
    print(f"Label: {result['label']}")
    return result


if __name__ == "__main__":
    # test
    smiles = "c12c3c(N4CCN(C)CC4)c(F)cc1c(c(C(O)=O)cn2C(C)CO3)=O"
    # 分类任务测试
    # name = "BBBP"
    # target = "p_np"
    # print("\n\nClassification task test:")
    # test_prediction(name, target, smiles)
    # 分类多任务测试
    # name = "ClinTox"
    # print("\n\nClassification multitask test:")
    # target = "FDA_APPROVED"
    # test_prediction(name, target, smiles)
    
    # target = "CT_TOX"
    # test_prediction(name, target, smiles)
    # 回归任务测试
    name = "FreeSolv"
    target = "expt"
    print("\n\nRegression task test:")
    test_prediction(name, target, smiles)
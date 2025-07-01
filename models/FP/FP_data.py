import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from pubchemfp import GetPubChemFPs  # 从pubchemfp.py导入指纹生成函数


def smiles_to_fingerprint(smiles_list, fp_type='mixed'):
    """完全匹配原始MoleData的处理方式"""
    fp_list = []
    valid_smiles = []
    
    for smiles in smiles_list:
        # 关键：不要标准化，直接使用原始SMILES
        mol = Chem.MolFromSmiles(smiles)  # 与MoleData完全一致
        
        # 复制原始过滤逻辑
        if mol is None:
            print(f"过滤无效SMILES: {smiles}")
            continue
            
        # 生成指纹（与FPN类一致）
        if fp_type == 'mixed':
            fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))
            fp_phaErGfp = list(AllChem.GetErGFingerprint(
                mol, fuzzIncrement=0.3, maxPath=21, minPath=1
            ))
            fp_pubcfp = list(GetPubChemFPs(mol))
            fp = fp_maccs + fp_phaErGfp + fp_pubcfp
        else:
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        
        fp_list.append(fp)
        valid_smiles.append(smiles)  # 保留原始SMILES
    
    return torch.tensor(fp_list, dtype=torch.float32)
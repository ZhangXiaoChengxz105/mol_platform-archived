import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem
from rdkit.Chem import AllChem
from pubchemfp import GetPubChemFPs  # 从pubchemfp.py导入指纹生成函数

def smiles_to_fingerprint(smiles):
    """将SMILES转换为分子指纹"""
    try:
        # 标准化SMILES
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("无效的SMILES")
        fp = []
        # 生成PubChem混合指纹
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
        fp_pubcfp = GetPubChemFPs(mol)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        return fp
        
    except Exception as e:
        raise ValueError(f"指纹生成失败: {str(e)}")

def canonicalize_smiles(smiles):
    """标准化SMILES字符串"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None
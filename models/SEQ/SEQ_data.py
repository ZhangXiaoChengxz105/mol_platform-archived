import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem
from tokenizer.tokenizer import MolTranBertTokenizer


TOKENIZER = MolTranBertTokenizer(os.path.join(models_root, "SEQ", "bert_vocab.txt"))
def smiles_to_tokens(smiles):
    """安全处理SMILES编码"""
    try:
        # 编码处理
        canon_smi = canonicalize_smiles(smiles)
        if not canon_smi:
            raise ValueError("SMILES标准化失败")
        
        # 直接使用原始tokenizer（非SEQ封装）
        tokenizer = TOKENIZER
        tokens = tokenizer.batch_encode_plus(
            [smiles] if isinstance(smiles, str) else smiles,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        data = [input_ids, attention_mask]
        return data
        
    except Exception as e:
        raise ValueError(f"SMILES编码失败: {str(e)}")


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None
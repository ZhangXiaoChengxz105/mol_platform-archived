import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT

# 使用与训练相同的常量定义（确保特征编码一致）
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def smiles_to_graph(smiles_list):
    """将SMILES字符串转换为图数据结构（预测专用）"""
    data_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 添加氢原子（保持与训练一致）
        mol = Chem.AddHs(mol)
        
        # 原子特征处理
        atom_features = []
        for atom in mol.GetAtoms():
            # 原子类型映射
            atomic_num = atom.GetAtomicNum()
            try:
                type_idx = ATOM_LIST.index(atomic_num)
            except ValueError:
                type_idx = len(ATOM_LIST) - 1  # 未知原子映射到最后一项
            
            # 手性特征映射
            try:
                chirality_idx = CHIRALITY_LIST.index(atom.GetChiralTag())
            except ValueError:
                chirality_idx = 0  # 未知手性映射到CHI_UNSPECIFIED
            
            atom_features.append([type_idx, chirality_idx])
        
        # 边特征处理
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # 键类型映射
            try:
                bond_type_idx = BOND_LIST.index(bond.GetBondType())
            except ValueError:
                bond_type_idx = 0
            
            # 键方向映射
            try:
                bond_dir_idx = BONDDIR_LIST.index(bond.GetBondDir())
            except ValueError:
                bond_dir_idx = 0
            
            # 双向添加（保持图的无向性）
            row += [start, end]
            col += [end, start]
            edge_feat.append([bond_type_idx, bond_dir_idx])
            edge_feat.append([bond_type_idx, bond_dir_idx])
        
        # 创建数据对象
        data =  Data(
            x=torch.tensor(atom_features, dtype=torch.long),
            edge_index=torch.tensor([row, col], dtype=torch.long),
            edge_attr=torch.tensor(edge_feat, dtype=torch.long),
            # 预测不需要真实标签，但为了兼容性保留y属性
            y=None  # 虚拟标签
        )
        data_list.append(data)

    return data_list
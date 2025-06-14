import torch
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    atom_features = []
    for atom in mol.GetAtoms():
        # 原子类型：确保在0-118范围内
        atomic_num = atom.GetAtomicNum()
        if atomic_num >= 119:
            atomic_num = 0  # 使用0作为unknown token
            
        # 手性信息：确保在0-2范围内  
        try:
            chiral_tag = int(atom.GetChiralTag())
            if chiral_tag >= 3:
                chiral_tag = 0
        except:
            chiral_tag = 0
            
        features = [atomic_num, chiral_tag]
        atom_features.append(features)
    
    # 边特征处理
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = int(bond.GetBondType())
        bond_dir = int(bond.GetBondDir())
        
        # 确保键类型和方向在有效范围内
        if bond_type >= 5: bond_type = 0
        if bond_dir >= 3: bond_dir = 0
        
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([[bond_type, bond_dir], [bond_type, bond_dir]])
    
    return Data(
        x=torch.tensor(atom_features, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long)
    )

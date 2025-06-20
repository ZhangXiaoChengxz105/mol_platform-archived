## Directory Structure

```
mol_platform/
└── dataset/
    ├── base.py                
    ├── provider.py            
    ├── squence.py             
    ├── smile_config.yaml      
    └── test_sequence_provider.py  
```

## Config Format (`smile_config.yaml`)

```yaml
datasets:
  BBBP:
    smiles_col: smiles
    label_cols: ["p_np"]
  Tox21:
    smiles_col: smiles
    label_cols: ["NR-AR", "NR-AR-LBD", ...]
```

## Dependencies

- Python ≥ 3.8
- [RDKit](https://www.rdkit.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)


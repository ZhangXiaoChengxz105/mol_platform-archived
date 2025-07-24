## Dataset Format Requirements

All datasets **must** be in `.csv` format. Each row represents a molecule with the following minimal fields:

- A molecular representation (e.g., SMILES string)
- One or more task labels (e.g., bioactivity, toxicity)

### Example CSV Format

```csv
data,MUV-466,MUV-548
CCOC(=O)c1ccc(C(=O)OCC)cc1,1,0
CCN(CC)CCOC(=O)c1ccc(C(=O)O)cc1,0,1
```

- **`data`**: Required column that provides molecular input.
- **Task labels**: Can be one or more columns depending on the dataset.

---

## `datasets.yaml` 配置文件要求

平台通过 `datasets.yaml` 文件来解析每个数据集的输入字段与任务标签，文件应满足以下格式与字段要求：

### 必须字段

- `data_type`: 指定输入数据格式
- `config`: 每个数据集对应的任务标签字典
- `dataset_names`: 可用数据集名称列表
- `regression_datasets`: 在上述数据集中属于回归任务的名称列表（其他视为分类任务）

### 示例结构

```yaml
data_type: "smiles"

config:
  Tox21:
    - NR-AR
    - NR-AR-LBD
    ...
  ClinTox:
    - FDA_APPROVED
    - CT_TOX
  ...

dataset_names:
  - Tox21
  - ClinTox
  - MUV
  - SIDER
  - BBBP
  - HIV
  - BACE
  - FreeSolv
  - ESOL
  - Lipo
  - qm7
  - qm8
  - qm9

regression_datasets:
  - FreeSolv
  - ESOL
  - Lipo
  - qm7
  - qm8
  - qm9
```


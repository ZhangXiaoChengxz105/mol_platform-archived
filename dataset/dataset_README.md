# 数据集格式要求

所有数据集必须为 `.csv` 格式。每一行表示一个分子，且必须包含以下字段：

- 分子表示（例如 SMILES 字符串）
- 一个或多个任务标签（例如生物活性、有毒性、溶解度等）

## 示例 CSV 格式

```csv
data,MUV-466,MUV-548
CCOC(=O)c1ccc(C(=O)OCC)cc1,1,0
CCN(CC)CCOC(=O)c1ccc(C(=O)O)cc1,0,1
```

- **`data`**：这是必须字段，通常填写 SMILES 字符串，表示分子结构。
- **任务标签**：一个或多个列，用于回归或分类任务。支持多任务学习场景。

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


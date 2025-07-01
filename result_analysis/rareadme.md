
# 📘 使用说明：模型运行与评估配置

为确保正常运行流程，请 **首先完成环境依赖安装**，并按以下步骤进行操作。

---

## 📦 1. 环境配置

请先执行 [`environment.md`](./environment.md) 中的命令以安装所需依赖环境。

---

## ⚙️ 2. 修改运行配置

配置文件路径：`config_run.yaml`  
通过编辑该文件可灵活控制模型的运行行为：

| 字段名称       | 说明 |
|----------------|------|
| `model`        | 指定使用的模型名称。模型位于 `models/` 文件夹下，可供参考。 |
| `name`         | 数据集名称。所有可用数据集名称在 `models/utils.py` 中可查。<br>可设为 `all` 对所有数据集依次运行。 |
| `eval`         | 是否启用评估模式。设为 `true` 时，将绘制图像并统计模型性能（如准确率、MSE、MAE 等）。 |
| `target_list`  | 需评估的任务目标（标签名）。<br>设为 `all` 可自动抓取该模型支持的全部任务（定义于 `models/utils.py`）。<br>多个目标以英文逗号分隔。 |
| `smiles_list`  | 需评估的分子（SMILES 格式）。支持以下三种形式：<ul><li>`all`：评估所有分子</li><li>`randomN`：如 `random200`，随机抽样 N 个分子</li><li>指定的 SMILES 字符串，多个用英文逗号分隔</li></ul> |
| `output`       | 结果 CSV 文件的保存文件夹路径。 |
| `plotpath`     | 图像文件的保存路径（文件夹）。 |

---

## 🧾 3. 输出说明

### 📄 CSV 文件命名方式

- 格式：`model_dataset_task.csv`  
- 示例：`fp_BBBP_p_np.csv`

---

### 📊 图像文件命名方式

#### ✅ 单个任务图（每个模型-数据集-目标）

- 分类任务（Classification）：  
  `model_dataset_task_classification.png`

- 回归任务（Regression）：  
  `model_dataset_task_regression.png`

#### 📈 整个数据集聚合图（同模型 & 同数据集的多个任务聚合）

- 分类聚合图：  
  `model_dataset_classification.png`

- 回归聚合图：  
  `model_dataset_regression.png`

#### 🌐 模型全局聚合图（跨多个数据集的所有任务）

- 分类聚合图：  
  `model_classification.png`

- 回归聚合图：  
  `model_regression.png`

---




from rdkit import Chem
import pandas as pd
import yaml
import os
from typing import List, Dict

class BaseDataset:
    def __init__(self, datasetname: str, datasetpath: str):
        self.datasetname = datasetname
        self.datasetpath = datasetpath
        self.data = None

    def loadData(self):
        df = pd.read_csv(self.datasetpath)
        self.data = df

    def preprocessData(self):
        pass

    def get_smiles_and_labels_by_config(
            self,
            model_type: str,
            task: str,
            config_file: str = "moleculnet_config.yaml"
    ) -> Dict[str, List]:
        """
            参数:
                model_type (str): 模型类别名，例如 'molecule_net'，对应 YAML 的一级键。
                task (str): 任务类型，例如 'classification' 或 'regression'，对应 YAML 的三级键。
                config_file (str): YAML 配置文件的路径，默认值为 "moleculnet_config.yaml"。

            返回:
                Dict[str, List]，包含两个键：
                    - "smiles": List[str]，所有合法 SMILES 字符串；
                    - "label": List[Any]，每个 SMILES 对应的标签，可能是标量（单任务）或列表（多任务）。
            """
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_file)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            task_cfg = config[model_type][self.datasetname][task]
        except KeyError:
            raise ValueError(f"配置项不存在: {model_type}/{self.datasetname}/{task}")

        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        smiles_list = []
        label_list = []

        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            if Chem.MolFromSmiles(smiles) is not None:
                smiles_list.append(smiles)
                label_list.append(label)

        return {
            "data": smiles_list,
            "label": label_list
        }

    def get_entry_by_smiles(
            self,
            smiles_str: str,
            target_col: str,
            model_type: str,
            task_type: str,
            config_file: str = "moleculnet_config.yaml"
    ):
        """
        查找指定 SMILES 对应的数据字段和目标字段的值。

        参数:
            smiles_str (str): 要查找的 SMILES 字符串
            target_col (str): 想要获取的目标字段（需在对应 task 的 label_cols 中）
            model_type (str): 如 "molecule_net"
            task_type (str): 如 "classification"
            config_file (str): 配置文件路径

        返回:
            (data_val, target_val): 数据字段值 和 目标字段值
        """
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        # === 读取配置 ===
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_file)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            task_cfg = config[model_type][self.datasetname][task_type]
        except KeyError:
            raise ValueError(f"配置项不存在: {model_type}/{self.datasetname}/{task_type}")

        smiles_col = task_cfg["smiles_col"]
        label_cols = task_cfg["label_cols"]

        if target_col not in label_cols:
            raise ValueError(f"目标列 '{target_col}' 不在配置的 label_cols 中: {label_cols}")

        # === 查找 ===
        match = self.data[self.data[smiles_col] == smiles_str]

        if match.empty:
            print(f"找不到 SMILES: {smiles_str}")
            return None, None

        data_value = match.iloc[0][smiles_col]
        target_value = match.iloc[0][target_col]

        return data_value, target_value

    def get_all_smiles_and_task_labels(
            self,
            model_type: str,
            config_file: str = "moleculnet_config.yaml"
    ) -> Dict[str, List]:
        """
        返回当前数据集中所有合法 SMILES 以及所有任务对应的标签列名。

        参数:
            model_type (str): 模型类型，如 'molecule_net'
            config_file (str): YAML 配置路径

        返回:
            dict:
                {
                    "smiles": [...],           # 所有合法 SMILES
                    "tasks": {
                        "classification": [...],   # 分类任务标签列名
                        "regression": [...],       # 回归任务标签列名
                        ...
                    }
                }
        """
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_file)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            dataset_config = config[model_type][self.datasetname]
        except KeyError:
            raise ValueError(f"配置项不存在: {model_type}/{self.datasetname}")

        tasks_label_names = {}
        valid_smiles_set = set()

        for task_name, task_cfg in dataset_config.items():
            smiles_col = task_cfg["smiles_col"]
            label_cols = task_cfg["label_cols"]

            tasks_label_names[task_name] = label_cols

            for _, row in self.data.iterrows():
                smiles = row[smiles_col]
                if Chem.MolFromSmiles(smiles) is not None:
                    valid_smiles_set.add(smiles)

        return {
            "data": list(valid_smiles_set),
            "tasks": tasks_label_names
        }

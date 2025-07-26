import os
import yaml
import pandas as pd
from typing import List, Dict


class BaseDataset:
    def __init__(self, datasetname: str, datasetpath: str):
        self.datasetname = datasetname
        self.datasetpath = datasetpath
        self.data = None

    def loadData(self):
        self.data = pd.read_csv(self.datasetpath)

    def _get_config(self, config_file: str = None) -> dict:
        # 默认使用固定路径 dataset/data/moleculenet/dataset.yaml
        if config_file is None:
            config_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                "dataset", "data", "moleculenet", "dataset.yaml"
            ))
        else:
            config_path = os.path.abspath(config_file)

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_data_and_labels_by_config(
        self,
        config_file: str = None
    ) -> Dict[str, List]:
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        config = self._get_config(config_file)

        if self.datasetname not in config["config"]:
            raise ValueError(f"未在配置中找到数据集 {self.datasetname}")

        label_cols = config["config"][self.datasetname]
        data_col = config.get("data_type", "smiles")

        data_list = []
        label_list = []

        for _, row in self.data.iterrows():
            datum = row[data_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else list(row[label_cols])

            data_list.append(datum)
            label_list.append(label)

        return {
            "data": data_list,
            "label": label_list
        }

    def get_entry_by_data(
        self,
        data_str: str,
        target_col: str,
        config_file: str = None
    ):
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        config = self._get_config(config_file)

        if self.datasetname not in config["config"]:
            raise ValueError(f"未在配置中找到数据集 {self.datasetname}")

        label_cols = config["config"][self.datasetname]
        data_col = config.get("data_type", "smiles")

        if target_col not in label_cols:
            raise ValueError(f"目标列 '{target_col}' 不在标签列中: {label_cols}")

        match = self.data[self.data[data_col] == data_str]
        if match.empty:
            print(f"找不到数据: {data_str}")
            return None, None

        return match.iloc[0][data_col], match.iloc[0][target_col]

    def get_all_data_and_task_labels(
        self,
        config_file: str = None
    ) -> Dict[str, List]:
        if self.data is None:
            raise ValueError("数据未加载，请先调用 loadData()")

        config = self._get_config(config_file)
        data_col = config.get("data_type", "smiles")

        valid_data_set = set()
        tasks_label_names = {}

        for task_name, label_cols in config["config"].items():
            tasks_label_names[task_name] = label_cols

            if task_name != self.datasetname:
                continue

            for _, row in self.data.iterrows():
                datum = row[data_col]
                valid_data_set.add(datum)

        return {
            "data": list(valid_data_set),
            "tasks": tasks_label_names
        }

    def get_task_labels_by_dataset(
            self,
            dataset_name: str = None,
            config_file: str = None
    ) -> List[str]:

        if dataset_name is None:
            dataset_name = self.datasetname

        config = self._get_config(config_file)
        task_config = config.get("config", {})

        if dataset_name not in task_config:
            raise ValueError(f"配置中找不到数据集 {dataset_name} 的标签信息")

        return task_config[dataset_name]

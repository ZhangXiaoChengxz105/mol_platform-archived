import yaml
import os

class CheckUtils:
    def __init__(self, name = "moleculenet"):
        """
        初始化CheckUtils类。
        
        参数:
            config_path: 配置文件的路径，如果为None，则使用默认配置
        """

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name, "check_utils.yaml")
        self._load_config()
    
    def _load_config(self):
        """
        从配置文件加载配置。
        
        参数:
        """

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 提取配置数据
        self.config = config_data.get('config', {})
        self.dataset_names = config_data.get('dataset_names', [])
        self.regression_tasks = config_data.get('regression_tasks', [])
    
    def get_config(self):
        """
        获取配置字典。
        
        返回:
            包含所有数据集配置的字典
        """
        return self.config
    
    def get_datasets(self):
        """
        获取所有数据集名称。
        
        返回:
            数据集名称列表
        """
        return self.dataset_names
    
    def get_datasets_measure_names(self, dataset_name):
        """
        获取MoleculeNet数据集的目标列表。
        
        参数:
            dataset_name: 数据集名称
            
        返回:
            对应数据集的目标列表
            
        异常:
            ValueError: 如果提供了无效的数据集名称
        """
        try:
            return self.config[dataset_name]
        except KeyError:
            raise ValueError(f"无效的数据集名称: {dataset_name}, 获取对应属性失败\n支持的数据集名称: {self.dataset_names}")
    
    def get_datasets_measure_numbers(self, dataset_name):
        """
        获取数据集目标的数量。
        
        参数:
            dataset_name: 数据集名称
            
        返回:
            目标数量
        """
        return len(self.get_datasets_measure_names(dataset_name))
    
    def get_datasets_task_type(self, dataset_name):
        """
        获取数据集的任务类型（回归或分类）。
        
        参数:
            dataset_name: 数据集名称
            
        返回:
            'regression'或'classification'
        """
        return 'regression' if dataset_name in self.regression_tasks else 'classification'
    
    def validate_datasets_measure_names(self, dataset_name, measure_name):
        """
        验证数据集和目标名称是否有效。
        
        参数:
            dataset_name: 数据集名称
            measure_name: 目标名称
            
        异常:
            ValueError: 如果数据集名称或目标名称无效
        """
        if dataset_name not in self.dataset_names:
            raise ValueError(f"无效的数据集名称: {dataset_name}, 获取对应属性失败\n支持的数据集名称: {self.dataset_names}")
        if measure_name not in self.get_datasets_measure_names(dataset_name):
            raise ValueError(f"数据集{dataset_name}无效的目标名称: {measure_name}, 获取对应属性失败\n支持的目标名称: {self.get_datasets_measure_names(dataset_name)}")

# 为了向后兼容，提供全局实例
default_utils = CheckUtils()

# 为了向后兼容，提供全局函数
def get_config():
    return default_utils.get_config()

def get_datasets():
    return default_utils.get_datasets()

def get_datasets_measure_names(dataset_name):
    return default_utils.get_datasets_measure_names(dataset_name)

def get_datasets_measure_numbers(dataset_name):
    return default_utils.get_datasets_measure_numbers(dataset_name)

def get_datasets_task_type(dataset_name):
    return default_utils.get_datasets_task_type(dataset_name)

def validate_datasets_measure_names(dataset_name, measure_name):
    return default_utils.validate_datasets_measure_names(dataset_name, measure_name)
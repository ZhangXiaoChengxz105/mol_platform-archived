import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
from abc import ABC, abstractmethod
import yaml
from utils import plot_csv_by_task
project_root = os.path.dirname(os.path.dirname(__file__))
provider_dir = os.path.join(project_root, 'dataset')
model_dir = os.path.join(project_root, 'models')
sys.path.append(provider_dir)
sys.path.append(model_dir)
from check_utils import validate_datasets_measure_names
import numpy as np
import torch
from base import BaseDataset
import re
import random
from collections import defaultdict
import csv
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class model_runner_interface(ABC):
    @abstractmethod
    def run(self):
        pass

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class Runner(model_runner_interface):
    def __init__(self, userarg,model, name, target_list, smiles_list, Model_type=None):
        self.model = model
        self.name = name
        self.target_list = target_list
        self.smiles_list = smiles_list
        self.Model_type = Model_type
        self.userarg = userarg

    def run(self):
        model_id = self.model.strip().lower()
        model_path = os.path.join(project_root, 'models',self.userarg, model_id.upper())

        if model_path not in sys.path:
            sys.path.append(model_path)

        import_stmt = f"from {model_id.upper()}_output import predict as predict_func"
        try:
            exec(import_stmt, globals())
        except Exception as e:
            print(f"❌ 模型导入失败: {e}")
            sys.exit(1)

        results = []
        for target in self.target_list:
            print(f"[DEBUG] calling predict_func with: name={self.name}, target={target}, Model_type={self.Model_type} ")
            try:
                # ✅ 根据是否传入 Model_type 动态构造调用
                if self.Model_type is not None:
                    result = predict_func(self.name, target, self.smiles_list, model_type=self.Model_type)
                else:
                    result = predict_func(self.name, target, self.smiles_list)

                for item in result:
                    if self.Model_type:
                        item["model"] = f"{self.model}_{self.Model_type}"
                    else:
                        item["model"] = self.model
                    item["name"] = self.name
                    item["target"] = target

                results.extend(result)

            except Exception as e:
                results.append({
                    "model": f"{self.model}_{self.Model_type}" if self.Model_type else self.model,
                    "target": target,
                    "data": self.smiles_list,
                    "error": str(e),
                    "name": self.name
                })

        return results

        # if self.output:
        #     with open(self.output, 'w') as f:
        #         json.dump(results, f, indent=4)
        #     print(f"\n✅ All results saved to: {self.output}")
        # else:
        #     print("\nPrediction Results:")
        #     print(json.dumps(results, indent=4))
        return results


class DatasetRunner(model_runner_interface):
    def __init__(self, model, name, target_list, smiles_list, output=None):
        self.model = model
        self.name = name
        self.target_list = target_list
        self.smiles_list = smiles_list
        self.output = output

    def run(self):
        dataset_path = os.path.join(project_root, 'dataset', 'data', f'{self.name}.csv')
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset file not found: {dataset_path}")
            sys.exit(1)

        dataset = BaseDataset(self.name, dataset_path)
        dataset.loadData()
        alldata = dataset.provideData(self.name)
        print(json.dumps(alldata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified model runner")
    parser.add_argument('--model', type=str, required=True,
                        help="Model type (e.g., gnn, seq)")
    parser.add_argument('--name', type=str, required=True,
                        help="Dataset name (e.g., BBBP, QM9)")
    parser.add_argument('--eval', type=lambda x: x.lower() == 'true', default=False,
                    help="Whether to run in evaluation mode (True/False)")
    parser.add_argument('--target_list', type=str, required=True,
                        help="Comma-separated target list (e.g., p_np,logD)")
    parser.add_argument('--smiles_list', type=str, required=True,
                        help="Comma-separated SMILES list (e.g., C1CCCCC1,c1ccccc1O)")
    parser.add_argument('--output', type=str, help="Optional output JSON file path")
    parser.add_argument('--runner', type=str, choices=['gnn', 'dataset'], default='gnn',
                        help="Choose which runner to execute")
    parser.add_argument('--plotpath', type=str, default="plots",
                        help="Directory to save plotted images (default: 'plots')")
    parser.add_argument('--plotprevisousruns', type=lambda x: x.lower() == 'true', default=False,
                        help="Whether to plot previous runs (True/False)")
    parser.add_argument('--user_argument',type=str,help = 'user specified argument for model calssification')
    # parser.add_argument('--Model_type', type = str, required = False, help = 'specific arugment for fp model type')
    parser.add_argument('--regression_tasks',type= str,help ='list of user specified metrics for plotting regression tasks')
    parser.add_argument('--classification_tasks',type=str,help = 'list of user specified metrics for classification')
    
    return parser.parse_args()

# def lookup(item,data):
#     model = item['model']
#     smiles= item['data']
#     task = item['task']
#     name = item['name']
#     target= item['target']
#     prediction = item['prediction']
#     label= item['label']
#     name_index, indexcnt = lookupindex(name, target)
#     try:
#         smiles_index = data['smiles'].index(smiles)
#         item["truth"]= data['label'][smiles_index][name_index] if indexcnt > 1 else data['label'][smiles_index]
#         return item
#     except ValueError:
#         return item
    
    
# def lookupindex(model,name):
#     # 查找对应模型中的 task的索引id、
#     # 查找单个数据集中的某个任务在这个数据集中的索引
#     config_path = os.path.join(project_root, 'dataset','moleculnet_config.yaml')
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
    
#     datasets = config.get("datasets", {})
    
#     # 忽略大小写查找 model
#     matched_model = None
#     for key in datasets:
#         if key.lower() == model.lower():
#             matched_model = key
#             break
    
#     if matched_model is None:
#         raise ValueError(f"Model '{model}' not found in config (case-insensitive match failed).")
    
#     label_cols = datasets[matched_model].get("label_cols", [])
    
#     # 忽略大小写查找 task
#     matched_name = None
#     for i, label in enumerate(label_cols):
#         if label.lower() == name.lower():
#             matched_name = i
#             break
    
#     if matched_name is None:
#         raise ValueError(f"name '{name}' not found in model '{matched_model}' labels (case-insensitive match failed).")
    
#     return matched_name, len(label_cols)

# def get_all_targets_and_smiles(name,data):
#     config_path = os.path.join(project_root, 'dataset','moleculnet_config.yaml')
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#         datasets = config.get("datasets", {})
    
#     # 忽略大小写匹配 name
#     matched_name = None
#     for key in datasets:
#         if key.lower() == name.lower():
#             matched_name = key
#             break
    
#     if matched_name is None:
#         raise ValueError(f"No dataset config found for '{name}'.")
#     smiles_col = data['smiles']
#     label_cols = datasets[matched_name]["label_cols"]
    
    
#     return smiles_col, label_cols
    
# def get_all_datasets(model: str,):
#     # 设置路径
#     config_path = os.path.join(project_root, "models", "model_datasets.yaml")

#     # 判断文件是否存在
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"模型配置文件未找到: {config_path}")

#     # 读取 YAML 文件
#     with open(config_path, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     # 获取模型对应部分
#     model_section = config.get("models", {}).get(model)
#     if not model_section:
#         raise ValueError(f"模型 '{model}' 不存在于配置文件中")

#     # 返回 datasets 列表
#     return model_section.get("datasets", [])
    

# def get_all_models():
#     # 假设 project_root 已定义，例如：
#     # project_root = os.path.dirname(os.path.dirname(__file__))
#     config_path = os.path.join(project_root, 'models', 'model_datasets.yaml')

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"配置文件未找到: {config_path}")

#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)

#     model_dict = config.get("models", {})
#     return list(model_dict.keys())
    
def get_latest_run_num(output):
    path = os.path.join(project_root, 'results', output)
    os.makedirs(path, exist_ok=True)
    run_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and re.match(r"run\d+", d)]
    
    if not run_dirs:
        return "run1"

    # 提取 run 后面的数字并找最大
    run_nums = [int(re.findall(r'\d+', d)[0]) for d in run_dirs]
    next_run = max(run_nums) + 1

    return f"run{next_run}"
    
    
    
        
    


if __name__ == '__main__':
    args = parse_args()
    grouped_results = defaultdict(list)
    Model_list = [m.strip().lower() for m in args.model.split(',')]
    config_datasets_path = os.path.join(provider_dir,"data",args.user_argument,'dataset.yaml')
    finalres = []
    for model in Model_list:
        if "_" in args.model:
            model, model_type = model.split("_", 1)
            model = model.strip().lower()
            model_type = model_type.strip().upper()
        else:
            model = model.strip().lower()
            model_type = None
        names_list =[s.strip() for s in args.name.split(',')]
        for name in names_list:
            ds = None
            smiles_arg = args.smiles_list.strip().lower()
            if args.target_list =='all' or smiles_arg == "all" or re.match(r"random\d+", smiles_arg):  
                ds = BaseDataset(name, os.path.join(project_root, 'dataset', 'data',args.user_argument, f'{name}.csv'))
                ds.loadData()
                ret = ds.get_all_data_and_task_labels(config_datasets_path)
            if args.target_list =='all':
                target_list = ret['tasks'][name]
            else:
                target_list = [t.strip() for t in args.target_list.split(',')]
            
            valid_targets = []
            for target in target_list:
                try:
                    validate_datasets_measure_names(name, target)
                    valid_targets.append(target)
                except ValueError as e:
                    print(f"⚠️ {e} —— 已移除目标 '{target}'")
            target_list = valid_targets

            if not target_list:
                print(f"❌ 数据集 {name} 无合法 target,跳过该项")
                continue

            # smiles_list
            if smiles_arg == "all":
                smiles_list = ret['data']
            elif re.match(r"random\d+", smiles_arg):
                count = int(re.findall(r"\d+", smiles_arg)[0])
                available = len(ret['data'])
                actual_count = min(count, available)
                if actual_count < count:
                    print(f"⚠️ Requested random{count}, but only {available} SMILES available. Using {actual_count}.")
                smiles_list = random.sample(ret['data'], actual_count)
            else:
                smiles_list = [s.strip() for s in args.smiles_list.split(',')]
            if model_type:
                runner = Runner(args.user_argument,model, name, target_list, smiles_list,model_type)
                result = runner.run()
                for i in range(len(result)):
                    subresult = result[i]
                    if "error" not in subresult:
                        if ds:
                            d,truth = ds.get_entry_by_data(subresult['data'],subresult['target'],config_datasets_path)
                            if truth != None:
                                subresult['truth'] = truth
                        finalres.append(subresult)
                    else:
                        finalres.append(subresult)
            else:
                runner = Runner(model, name, target_list, smiles_list)
                result = runner.run()
                for i in range(len(result)):
                    subresult = result[i]
                    if "error" not in subresult:
                        if ds:
                            d,truth = ds.get_entry_by_data(subresult['data'],subresult['target'],config_datasets_path)
                            if truth != None:
                                subresult['truth'] = truth
                        finalres.append(subresult)
                    else:
                        finalres.append(subresult)
            
    if finalres:
        # ⏬ 按 model_name_target 分组
        for i, item in enumerate(finalres):
            if "name" not in item or item["name"] is None:
                print(f"❌ 缺少 'name' 的条目 #{i}: {item}")
        for item in finalres:
            base_key = f"{item['model']}_{item['name']}_{item['target']}"
            if 'error' in item and item['error']:
                key = f"{base_key}_error"
            else:
                key = base_key
            grouped_results[key].append(item)
        runid = get_latest_run_num(args.output) if args.output else "run1"

    output_dir = args.output if args.output else "output"
    all_output_dir = os.path.join(project_root, 'results', output_dir)
    output_dir = os.path.join(project_root, 'results', output_dir,runid)

    
    os.makedirs(output_dir, exist_ok=True)

    written_files = []

    for key, items in grouped_results.items():
        output_path = os.path.join(output_dir, f"{key}.csv")

        # 写入 CSV 文件
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = None
            for item in items:
                safe_item = make_json_safe(item)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(safe_item.keys()))
                    writer.writeheader()
                writer.writerow(safe_item)

        print(f"✅ Results written to {output_path}")
        written_files.append(output_path)
        
    def comma_separated_list(value):
        return [item.strip() for item in value.split(",") if item.strip()]
    regression_tasks = [s.strip() for s in args.regression_tasks.split(",") if s.strip()]
    classification_tasks = [s.strip() for s in args.classification_tasks.split(",") if s.strip()]
    # ✅ 执行绘图（如果开启 eval 模式）
    if args.eval:
        if not args.plotprevisousruns:
            plot_csv_by_task(output_dir,regression_tasks,classification_tasks, save_dir=os.path.join(output_dir,args.plotpath))
        else:
            plot_csv_by_task(all_output_dir,regression_tasks,classification_tasks,save_dir=os.path.join(output_dir,args.plotpath))

    else:
        print("⚠️ No results to write.")
        
        # result = result[0]
        # print(result)
        # for item in result:
            
            
        
        
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
from squence import sequenceDataset
from base import BaseDataset
import re
import random
from collections import defaultdict
import csv


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
    def __init__(self, model, name, target_list, smiles_list, output=None):
        self.model = model
        self.name = name
        self.target_list = target_list
        self.smiles_list = smiles_list
        self.output = output

    def run(self):
        model_id = self.model.strip().lower()
        model_path = os.path.join(project_root, 'models', model_id.upper())

        if model_path not in sys.path:
            sys.path.append(model_path)

        import_stmt = f"from {model_id.upper()}_output import {model_id}_predict as predict_func"
        try:
            exec(import_stmt, globals())
        except Exception as e:
            print(f"❌ 模型导入失败: {e}")
            sys.exit(1)

        results = []
        for target in self.target_list:
            try:
                result = predict_func(self.name, target, self.smiles_list)
                for item in result:
                    item["model"]= self.model
                    item["name"]= self.name
                    item["target"] = target
                
                results.extend(result)
            except Exception as e:
                results.append({ 
                    "model": self.model,
                    "target": target,
                    "smiles": self.smiles_list,
                    "error": str(e),
                    "name": self.name
                })

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
    
    return parser.parse_args()

def lookup(item,data):
    model = item['model']
    smiles= item['smiles']
    task = item['task']
    name = item['name']
    target= item['target']
    prediction = item['prediction']
    label= item['label']
    name_index, indexcnt = lookupindex(name, target)
    try:
        smiles_index = data['smiles'].index(smiles)
        item["truth"]= data['label'][smiles_index][name_index] if indexcnt > 1 else data['label'][smiles_index]
        return item
    except ValueError:
        return item
    
    
def lookupindex(model,name):
    # 查找对应模型中的 task的索引id、
    config_path = os.path.join(project_root, 'dataset','smile_config.yaml')   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get("datasets", {})
    
    # 忽略大小写查找 model
    matched_model = None
    for key in datasets:
        if key.lower() == model.lower():
            matched_model = key
            break
    
    if matched_model is None:
        raise ValueError(f"Model '{model}' not found in config (case-insensitive match failed).")
    
    label_cols = datasets[matched_model].get("label_cols", [])
    
    # 忽略大小写查找 task
    matched_name = None
    for i, label in enumerate(label_cols):
        if label.lower() == name.lower():
            matched_name = i
            break
    
    if matched_name is None:
        raise ValueError(f"name '{name}' not found in model '{matched_model}' labels (case-insensitive match failed).")
    
    return matched_name, len(label_cols)

def get_all_targets_and_smiles(name,data):
    config_path = os.path.join(project_root, 'dataset','smile_config.yaml')   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        datasets = config.get("datasets", {})
    
    # 忽略大小写匹配 name
    matched_name = None
    for key in datasets:
        if key.lower() == name.lower():
            matched_name = key
            break
    
    if matched_name is None:
        raise ValueError(f"No dataset config found for '{name}'.")
    smiles_col = data['smiles']
    label_cols = datasets[matched_name]["label_cols"]
    
    
    return smiles_col, label_cols
    
def get_all_datasets(model: str):
    # 路径设置
    config_path = os.path.join(project_root, "dataset", "smile_config.yaml")
    model_dir = os.path.join(project_root, "models", f"{model}_finetune")

    # 读取 config 中的所有数据集名
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    all_dataset_names = config.get("datasets", {}).keys()

    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model path '{model_dir}' does not exist.")
    
    # 获取模型目录下所有文件名（不含扩展名，统一小写）
    all_files = os.listdir(model_dir)
    file_prefixes = [os.path.splitext(f)[0].lower() for f in all_files if os.path.isfile(os.path.join(model_dir, f))]

    # 匹配数据集名
    matched_names = []
    for dataset_name in all_dataset_names:
        dataset_lc = dataset_name.lower()
        if any(prefix.startswith(dataset_lc) for prefix in file_prefixes):
            matched_names.append(dataset_name)

    return matched_names

def get_all_models():
    # 路径设置
    model_dir = os.path.join(project_root, "models")

    # 获取模型目录下所有文件夹名（不含扩展名，统一小写）
    all_dirs = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and not d.endswith("finetune")
    ]
    return all_dirs
        
    


if __name__ == '__main__':
    args = parse_args()

    # 参数拆分处理
    if args.model.strip().lower() == "all":
        Model_list = get_all_models()
    else:
        Model_list = [m.strip().lower() for m in args.model.split(',')]
    for model in Model_list:
        if args.name.lower() == 'all':
            names_list  = get_all_datasets(model)
        else:
            names_list = [args.name]
        finalres = []
        for name in names_list:    
            ds = sequenceDataset(args.name, os.path.join(project_root, 'dataset', 'data', f'{name}.csv'))
            ds.loadData()
            data = ds.provideSmilesAndLabel(name)
            tmpsm, tmptg = get_all_targets_and_smiles(name, data)

            # target_list
            if args.target_list.strip().lower() == "all":
                target_list = tmptg
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
                print(f"❌ 数据集 {name} 无合法 target，跳过该项")
                continue

            smiles_arg = args.smiles_list.strip().lower()

            # smiles_list
            if smiles_arg == "all":
                smiles_list = tmpsm
            elif re.match(r"random\d+", smiles_arg):
                count = int(re.findall(r"\d+", smiles_arg)[0])
                available = len(tmpsm)
                actual_count = min(count, available)
                if actual_count < count:
                    print(f"⚠️ Requested random{count}, but only {available} SMILES available. Using {actual_count}.")
                smiles_list = random.sample(tmpsm, actual_count)
            else:
                smiles_list = [s.strip() for s in args.smiles_list.split(',')]
            runner = Runner(model, name, target_list, smiles_list, args.output)
            result = runner.run()
            for i in range(len(result)):
                subresult = result[i]
                if "error" not in subresult:
                    subresult = lookup(subresult,data)
                    finalres.append(subresult)
                else:
                    finalres.append(subresult)
            
        if finalres:
            # ⏬ 按 model_name_target 分组
            grouped_results = defaultdict(list)
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

            output_dir = args.output if args.output else "output"
            output_dir = os.path.join(project_root, 'results', output_dir)
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

            # ✅ 执行绘图（如果开启 eval 模式）
            if args.eval:
                plot_csv_by_task(output_dir, save_dir=os.path.join(output_dir,args.plotpath))

        else:
            print("⚠️ No results to write.")
        
        # result = result[0]
        # print(result)
        # for item in result:
            
            
        
        
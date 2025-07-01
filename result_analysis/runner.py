import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
from abc import ABC, abstractmethod
import yaml
from utils import plot_jsonl_by_task
runner_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(runner_dir, '..'))
provider_dir = os.path.join(project_root, 'dataset')
sys.path.append(provider_dir)
import numpy as np
import torch
from squence import sequenceDataset
from base import BaseDataset
import re
import random
from collections import defaultdict


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
                    "error": str(e)
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
    config_path = os.path.join(project_root,"dataset", "smile_config.yaml")
    model_dir = os.path.join(project_root,"models", f"{model}_finetune")

    # 读取 config 中的所有数据集名
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    all_dataset_names = config.get("datasets", {}).keys()

    # 获取模型目录下所有 .pt 文件名（不带后缀）
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model path '{model_dir}' does not exist.")
    
    pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    pt_prefixes = [os.path.splitext(f)[0].lower() for f in pt_files]

    # 找到交集：只要 pt 文件前缀包含 dataset 名，就认为匹配
    matched_names = []
    for dataset_name in all_dataset_names:
        dataset_lc = dataset_name.lower()
        if any(pt_prefix.startswith(dataset_lc) for pt_prefix in pt_prefixes):
            matched_names.append(dataset_name)

    return matched_names
        
    


if __name__ == '__main__':
    args = parse_args()

    # 参数拆分处理
    if args.name.lower() == 'all':
        names_list  = get_all_datasets(args.model)
    else:
        names_list = [args.name]
    for name in names_list:    
        ds = sequenceDataset(args.name, f"../dataset/data/{name}.csv")
        ds.loadData()
        data = ds.provideSmilesAndLabel(name)
        tmpsm, tmptg = get_all_targets_and_smiles(name, data)

        # target_list
        if args.target_list.strip().lower() == "all":
            target_list = tmptg
        else:
            target_list = [t.strip() for t in args.target_list.split(',')]

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
        
        finalres = []
        runner = Runner(args.model, name, target_list, smiles_list, args.output)
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
        for item in finalres:
            key = f"{item['model']}_{item['name']}_{item['target']}"
            grouped_results[key].append(item)

        output_dir = args.output if args.output else "output"
        os.makedirs(output_dir, exist_ok=True)

        written_files = []

        for key, items in grouped_results.items():
            output_path = os.path.join(output_dir, f"{key}.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for item in items:
                    safe_item = make_json_safe(item)
                    json.dump(safe_item, f, ensure_ascii=False)
                    f.write('\n')
            print(f"✅ Results written to {output_path}")
            written_files.append(output_path)

        # ✅ 执行绘图（如果开启 eval 模式）
        if args.eval:
            for path in written_files:
                plot_jsonl_by_task(path, save_dir=args.plotpath)
    else:
        print("⚠️ No results to write.")
        
        # result = result[0]
        # print(result)
        # for item in result:
            
            
        
        
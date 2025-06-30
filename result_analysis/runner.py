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

from squence import sequenceDataset
from base import BaseDataset


class model_runner_interface(ABC):
    @abstractmethod
    def run(self):
        pass


class GNNRunner(model_runner_interface):
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
        print(results)
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
    
        
    


if __name__ == '__main__':
    args = parse_args()

    # 参数拆分处理
    ds = sequenceDataset(args.name, f"../dataset/data/{args.name}.csv")
    ds.loadData()
    data = ds.provideSmilesAndLabel(args.name)
    tmpsm, tmptg = get_all_targets_and_smiles(args.name, data)

    # target_list
    if args.target_list.strip().lower() == "all":
        target_list = tmptg
    else:
        target_list = [t.strip() for t in args.target_list.split(',')]

    # smiles_list
    if args.smiles_list.strip().lower() == "all":
        smiles_list = tmpsm
    else:
        smiles_list = [s.strip() for s in args.smiles_list.split(',')]
    
    finalres = []
    runner = GNNRunner(args.model, args.name, target_list, smiles_list, args.output)
    result = runner.run()
    for i in range(len(result)):
        subresult = result[i]
        print(subresult)
        if "error" not in subresult:
            subresult = lookup(subresult,data)
            print(subresult)
            finalres.append(subresult)
        else:
            finalres.append(subresult)
            
    if finalres:
        output_file = args.output if args.output else "results.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for item in finalres:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ Results written to {output_file} in JSONL format")
    else:
        print("⚠️ No results to write.")
    if args.eval:
        plot_jsonl_by_task(output_file, save_dir=args.plotpath)
        
        # result = result[0]
        # print(result)
        # for item in result:
            
            
        
        
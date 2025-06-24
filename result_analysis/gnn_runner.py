import sys
import os
import argparse
import json
from abc import ABC, abstractmethod


class model_runner_interface(ABC):
    @abstractmethod
    def run(self):
        pass


class GNNRunner(model_runner_interface):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Flexible model runner")

        self.parser.add_argument('--model', type=str, required=True,
                                 help="Model type (e.g., gnn, seq)")
        self.parser.add_argument('--name', type=str, required=True,
                                 help="Dataset name (e.g., BBBP, QM9)")
        self.parser.add_argument('--target_list', type=str, required=True,
                                 help="Comma-separated target list (e.g., p_np,logD,expt)")
        self.parser.add_argument('--smiles_list', type=str, required=True,
                                 help="Comma-separated SMILES list (e.g., C1CCCCC1,c1ccccc1O)")
        self.parser.add_argument('--output', type=str, help="Optional output JSON file path")

    def run(self):
        args = self.parser.parse_args()

        # Step 1: 拼接模型路径：new_mol_platform/models/GNN/
        model_id = args.model.strip().lower()
        runner_dir = os.path.dirname(__file__)  # 当前是 result_analysis/
        project_root = os.path.abspath(os.path.join(runner_dir, '..'))  # new_mol_platform/
        model_path = os.path.join(project_root, 'models', model_id.upper())  # models/GNN/

        if model_path not in sys.path:
            sys.path.append(model_path)

        # Step 2: 使用 exec 执行固定格式的导入语句
        import_stmt = f"from {model_id.upper()}_output import {model_id}_predict as predict_func"
        try:
            exec(import_stmt, globals())
        except Exception as e:
            print(f"❌ 模型导入失败: {e}")
            sys.exit(1)

        # Step 3: 参数解析
        target_list = [t.strip() for t in args.target_list.split(',')]
        smiles_list = [s.strip() for s in args.smiles_list.split(',')]

        results = []
        for target in target_list:
            for smiles in smiles_list:
                try:
                    result = predict_func(args.name, target, smiles)
                    result["target"] = target
                    results.append(result)
                except Exception as e:
                    results.append({
                        "smiles": smiles,
                        "target": target,
                        "error": str(e)
                    })

        # Step 4: 输出结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✅ All results saved to: {args.output}")
        else:
            print("\nPrediction Results:")
            print(json.dumps(results, indent=4))


if __name__ == '__main__':
    runner = GNNRunner()
    runner.run()
# 这是对针对 gnn_ouput 的 调用接口
import sys
import os
import argparse
import json
from abc import ABC, abstractmethod

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/GNN'))
if module_path not in sys.path:
    sys.path.append(module_path)

from GNN_output import gnn_predict


class model_runner_interface(ABC):
    @abstractmethod
    def run(self):
        pass


class GNNRunner(model_runner_interface):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Flexible GNN model runner")

        self.parser.add_argument('--name', type=str, required=True, help="Model name (e.g., BBBP, FreeSolv)")
        self.parser.add_argument('--target_list', type=str, required=True,
                                 help="Comma-separated target list (e.g., p_np,logD,expt)")
        self.parser.add_argument('--smiles_list', type=str, required=True,
                                 help="Comma-separated SMILES list (e.g., C1CCCCC1,c1ccccc1O)")
        self.parser.add_argument('--output', type=str, help="Optional output JSON file")

    def run(self):
        args = self.parser.parse_args()

        target_list = [t.strip() for t in args.target_list.split(',')]
        smiles_list = [s.strip() for s in args.smiles_list.split(',')]

        results = []

        for target in target_list:
            for smiles in smiles_list:
                try:
                    result = gnn_predict(args.name, target, smiles)
                    result["target"] = target  # 添加 target 字段
                    results.append(result)
                except Exception as e:
                    results.append({
                        "smiles": smiles,
                        "target": target,
                        "error": str(e)
                    })

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
import os
import sys
import yaml

# Step 1: 加载配置
config_path = os.path.join(os.path.dirname(__file__), "config_run.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Step 2: 构造 sys.argv 模拟命令行输入
script_path = os.path.join(os.path.dirname(__file__), "runner.py")
sys.argv = [script_path]

# 添加参数（key=value 转换为 CLI 参数）
for k, v in config.items():
    sys.argv += [f"--{k}", str(v)]

# Step 3: 执行 gnn_runner.py 脚本
exec(open(script_path, encoding="utf-8").read())
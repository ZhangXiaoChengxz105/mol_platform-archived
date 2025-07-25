import subprocess
import sys
import os
from datetime import datetime

ENV_NAME = "molplat14"
PYTHON_VERSION = "3.11"

REQUIREMENTS_LIST = [
    "requirements.txt",
]

FULL_REQUIREMENTS_FILE = "requirements_full.txt"
NO_VERSION_FILE = "requirements.txt"

def run(cmd, shell=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"❌ 命令失败: {cmd}")
        sys.exit(1)

def conda_env_exists(env_name):
    result = subprocess.run(f"conda env list", shell=True, stdout=subprocess.PIPE, text=True)
    return any(line.startswith(env_name + " ") or line.endswith(env_name) for line in result.stdout.splitlines())
def install_requirements():
    for req_file in REQUIREMENTS_LIST:
        print(f"📦 安装依赖文件: {req_file}")
        run(f"conda run -n {ENV_NAME} python -m pip install -r {req_file}")
def update():
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE)
    except Exception:
        print("❌ 错误：找不到 conda，请先安装并确保已加入 PATH")
        sys.exit(1)

    if not conda_env_exists(ENV_NAME):
        print(f"📦 创建 conda 环境: {ENV_NAME}（Python {PYTHON_VERSION}）")
        run(f"conda create -y -n {ENV_NAME} python={PYTHON_VERSION}")
    else:
        print(f"✅ 环境 {ENV_NAME} 已存在，跳过创建")

    install_requirements()

    print(f"\n✅ 环境 {ENV_NAME} 已更新并安装依赖。")

def export():
    try:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "base")
        print(f"📤 正在导出 pip 包，当前环境: {env_name}")

        pip_freeze = subprocess.check_output(["pip", "freeze"], text=True).splitlines()

        pkgs_full = []
        pkgs_noversion = []

        for line in pip_freeze:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-e '):
                continue
            pkgs_full.append(line)
            pkg_name = line.split('==')[0].split('>=')[0].split('>')[0].split('<=')[0].split('<')[0].strip()
            pkgs_noversion.append(pkg_name)

        with open(FULL_REQUIREMENTS_FILE, "w", encoding='utf-8') as f:
            f.write("\n".join(pkgs_full))
        print(f"💾 已保存完整 requirements: {FULL_REQUIREMENTS_FILE}")

        with open(NO_VERSION_FILE, "w", encoding='utf-8') as f:
            f.write("\n".join(pkgs_noversion))
        print(f"💾 已保存无版本号 requirements: {NO_VERSION_FILE}")

        return True
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("⚠️ 请输入操作命令： update 或 export")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "update":
        update()
    elif cmd == "export":
        export()
    else:
        print(f"❌ 未知命令: {cmd}，可用命令为 update 或 export")
        sys.exit(1)

if __name__ == "__main__":
    main()
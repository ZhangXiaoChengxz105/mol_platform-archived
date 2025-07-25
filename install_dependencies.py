import subprocess
import sys
import os

ENV_NAME = "molplat4"   # 你可以修改名字
PYTHON_VERSION = "3.11"

# 这里维护所有 requirements.txt 路径，可以根据需要扩展
REQUIREMENTS_LIST = [
    "platform_requirements.txt",
]

def run(cmd, shell=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"命令失败: {cmd}")
        sys.exit(1)

def main():
    # 检查 conda 是否存在
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE)
    except Exception:
        print("错误：找不到 conda，请先安装并确保已加入 PATH")
        sys.exit(1)

    print(f"创建 conda 环境: {ENV_NAME}（Python {PYTHON_VERSION}）")
    run(f"conda create -y -n {ENV_NAME} python={PYTHON_VERSION}")

    print("开始安装 requirements 列表中的依赖...")

    if os.name == "nt":  # Windows PowerShell
        conda_base = os.path.join(os.environ['USERPROFILE'], "anaconda3")
        conda_hook = os.path.join(conda_base, "shell", "condabin", "conda-hook.ps1")

        # PowerShell 脚本多条命令合并
        # 用;分割多条 pip install 命令
        pip_install_cmds = "; ".join([f"pip install -r '{req}'" for req in REQUIREMENTS_LIST])
        
        powershell_cmd = f"""
        powershell -NoProfile -Command "& {{
            . '{conda_hook}';
            conda activate {ENV_NAME};
            {pip_install_cmds}
        }}"
        """
        run(powershell_cmd)
    else:
        # Linux/macOS bash
        conda_base = subprocess.check_output("conda info --base", shell=True).decode().strip()
        pip_install_cmds = " && ".join([f"pip install -r '{req}'" for req in REQUIREMENTS_LIST])
        bash_cmd = f"bash -c 'source {conda_base}/etc/profile.d/conda.sh && conda activate {ENV_NAME} && {pip_install_cmds}'"
        run(bash_cmd)

    print(f"\n完成！环境 {ENV_NAME} 已创建并安装依赖。")

    # 提示用户如何激活环境，脚本内激活不会影响外部shell
    print("\n请在你的终端执行以下命令激活环境：")
    if os.name == "nt":
        print(f"  conda activate {ENV_NAME}")
    else:
        print(f"  source activate {ENV_NAME}   或   conda activate {ENV_NAME}")

if __name__ == "__main__":
    main()
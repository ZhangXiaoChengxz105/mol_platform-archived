import argparse
import subprocess
import sys
import os
import platform
import re
import datetime
from pathlib import Path

# 全局默认配置
DEFAULT_ENV_NAME = "molplat"
DEFAULT_PYTHON_VERSION = "3.11.8"
DEFAULT_PIP_FILE = "requirements.txt"

def get_system_encoding():
    """获取系统默认编码"""
    try:
        if platform.system() == "Windows":
            return "utf-8"
        return sys.getdefaultencoding() or "utf-8"
    except:
        return "utf-8"

SYSTEM_ENCODING = get_system_encoding()

def run_command_realtime(cmd):
    """运行命令并实时输出到终端"""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding=SYSTEM_ENCODING,
            errors="replace",
            bufsize=1,
            shell=platform.system() == "Windows",
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f">>> {output.strip()}")

        stderr = process.stderr.read()
        if stderr:
            print(f"!!! {stderr.strip()}")

        return process.returncode

    except Exception as e:
        print(f"❌ 执行命令失败: {str(e)}")
        return -1

def get_current_env_name():
    """获取当前激活的环境名称"""
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if env_name:
        return env_name.split(os.sep)[-1]

    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "*" in line:
                    parts = line.split()
                    return parts[-1] if len(parts) > 1 else parts[0]
    except:
        pass

    try:
        result = subprocess.run(
            ["pip", "-V"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )
        if result.returncode == 0 and "site-packages" in result.stdout:
            match = re.search(r"/(\w+)/lib/python", result.stdout)
            if match:
                return match.group(1)
    except:
        pass

    return None

def export_environment(output_file):
    """导出当前环境的pip依赖到指定文件"""
    try:
        env_name = get_current_env_name()
        if not env_name:
            print("❌ 无法确定当前激活的环境")
            print("💡 请确保在Conda环境中运行此命令")
            return False

        print(f"📤 正在导出环境: {env_name}")
        print(f"📝 输出文件: {output_file}")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        result = subprocess.run(
            ["pip", "list", "--not-required", "--format=freeze"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )

        if result.returncode != 0:
            print(f"❌ 获取安装包失败: {result.stderr}")
            return False

        user_packages = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.startswith(("-e", "@", "#")):
                if ";" in line:
                    line = line.split(";")[0].strip()
                user_packages.append(line)

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(user_packages))

        print(f"✅ Pip依赖已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_environment(requirements_file):
    """根据指定的requirements.txt创建新环境"""
    try:
        req_file = Path(requirements_file)
        if not req_file.exists():
            print(f"❌ 错误: {req_file} 文件不存在")
            return False

        env_name = input("请输入新环境名称: ").strip()
        if not env_name:
            print("❌ 环境名称不能为空")
            return False

        python_version = input("请输入Python版本 (例如 3.11.8): ").strip()
        if not re.match(r"\d+\.\d+\.\d+", python_version):
            print("❌ 无效的Python版本格式")
            return False

        print(f"🛠️ 正在创建环境 '{env_name}'...")
        print("=" * 80)

        return_code = run_command_realtime(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"]
        )

        if return_code != 0:
            print(f"\n❌ 环境创建失败 (返回码: {return_code})")
            return False

        env_path = get_conda_env_path(env_name)
        if not env_path:
            print("\n❌ 无法找到环境路径")
            return False

        pip_exec = "pip.exe" if platform.system() == "Windows" else "pip"
        pip_path = os.path.join(env_path, "bin", pip_exec) if platform.system() != "Windows" else os.path.join(env_path, "Scripts", pip_exec)
        
        if not os.path.exists(pip_path):
            print(f"\n❌ 找不到pip可执行文件: {pip_path}")
            return False

        print(f"📦 正在安装依赖...")
        print("=" * 80)

        return_code = run_command_realtime(
            [pip_path, "install", "-r", str(req_file)]
        )

        print("=" * 80)

        if return_code == 0:
            print(f"\n✅ 环境 '{env_name}' 创建并配置成功!")
            print(f"👉 使用以下命令激活环境: conda activate {env_name}")
            return True
        else:
            print(f"\n❌ 依赖安装失败 (返回码: {return_code})")
            return False

    except Exception as e:
        print(f"⚠️ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_conda_env_path(env_name):
    """获取conda环境的完整路径"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )
        
        if result.returncode != 0:
            print(f"❌ 获取环境列表失败: {result.stderr}")
            return None
            
        for line in result.stdout.splitlines():
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == env_name:
                return parts[1]
                
        print(f"❌ 找不到环境: {env_name}")
        return None
        
    except Exception as e:
        print(f"❌ 获取环境路径失败: {str(e)}")
        return None

def update_environment(requirements_file):
    """使用指定的requirements.txt更新当前环境"""
    try:
        env_name = get_current_env_name()
        if not env_name:
            print("❌ 无法确定当前激活的环境")
            return False

        req_file = Path(requirements_file)
        if not req_file.exists():
            print(f"❌ 错误: {req_file} 文件不存在")
            return False

        print(f"🔄 正在更新当前环境 '{env_name}'...")
        print(f"📦 使用的依赖文件: {req_file}")
        print("=" * 80)

        return_code = run_command_realtime(
            ["pip", "install", "--upgrade", "-r", str(req_file)]
        )

        print("=" * 80)

        if return_code == 0:
            print("\n✅ 环境更新成功!")
            return True
        else:
            print(f"\n❌ 更新失败 (返回码: {return_code})")
            return False

    except Exception as e:
        print(f"⚠️ 发生错误: {str(e)}")
        return False

def main():
    # 主帮助信息
    parser = argparse.ArgumentParser(
        description="Python环境管理工具 - 简化Conda环境创建、导出和更新",
        epilog="使用示例:\n"
               "  导出环境: env_utils.py export (-r export_req.txt -e env_name -p python_version)\n"
               "  创建环境: env_utils.py create (-r create_req.txt)\n"
               "  更新环境: env_utils.py update (-r update_req.txt)\n"
               "  默认路径: -r requirements.txt"
               "  默认环境名：-e molplat"
               "  默认Python版本: -p 3.11.8",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", 
        title="可用命令",
    )

    # 导出命令
    export_parser = subparsers.add_parser(
        "export", 
        help="导出当前环境配置",
        description="导出当前环境的pip依赖到requirements.txt文件"
    )
    export_parser.add_argument(
        "-r", "--output", 
        default=DEFAULT_PIP_FILE,
        metavar="FILE",
        help=f"指定requirements.txt输出路径 (默认: {DEFAULT_PIP_FILE})"
    )
    export_parser.epilog = "示例: env_utils.py export -o myenv/requirements.txt"

    # 创建命令 - 新增环境名称和Python版本参数
    create_parser = subparsers.add_parser(
        "create", 
        help="创建新环境",
        description="根据requirements.txt创建新环境"
    )
    create_parser.add_argument(
        "-r", "--requirements", 
        default=DEFAULT_PIP_FILE,
        metavar="FILE",
        help=f"指定requirements.txt文件路径 (默认: {DEFAULT_PIP_FILE})"
    )
    create_parser.add_argument(
        "-e", "--env-name", 
        default=None,
        metavar="NAME",
        help=f"指定环境名称 (默认: {DEFAULT_ENV_NAME})"
    )
    create_parser.add_argument(
        "-p", "--python-version", 
        default=None,
        metavar="VERSION",
        help=f"指定Python版本 (默认: {DEFAULT_PYTHON_VERSION})"
    )
    create_parser.epilog = (
        "示例:\n"
        "  完全交互式: env_utils.py create\n"
        "  指定所有参数: env_utils.py create -r custom_req.txt -e myenv -p 3.11.8\n"
        "  仅指定依赖文件: env_utils.py create -r custom_req.txt"
    )

    # 更新命令
    update_parser = subparsers.add_parser(
        "update", 
        help="更新当前环境",
        description="使用requirements.txt更新当前环境"
    )
    update_parser.add_argument(
        "-r", "--requirements", 
        default=DEFAULT_PIP_FILE,
        metavar="FILE",
        help=f"指定requirements.txt文件路径 (默认: {DEFAULT_PIP_FILE})"
    )
    update_parser.epilog = "示例: env_utils.py update -r updated_requirements.txt"

    # 如果没有提供任何参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"🚀 执行命令: {args.command.upper()}")
    print("=" * 60)

    if args.command == "export":
        success = export_environment(args.output)
    elif args.command == "create":
        success = create_environment(args.requirements)
    elif args.command == "update":
        success = update_environment(args.requirements)
    else:
        print(f"❌ 未知命令: {args.command}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"{'✅ 操作成功' if success else '❌ 操作失败'}")
    print("=" * 60)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Python环境管理工具")
    print("=" * 60)
    main()
import argparse
import subprocess
import sys
import os
import platform
import re
import datetime
from pathlib import Path
import signal
import psutil  # 添加psutil用于进程树管理

# 全局默认配置
DEFAULT_ENV_NAME = "molplat"
DEFAULT_PYTHON_VERSION = "3.11.8"
DEFAULT_PIP_FILE = "requirements.txt"

# 全局变量跟踪活动进程
active_processes = []

def get_system_encoding():
    """获取系统默认编码"""
    try:
        if platform.system() == "Windows":
            return "utf-8"
        return sys.getdefaultencoding() or "utf-8"
    except:
        return "utf-8"

SYSTEM_ENCODING = get_system_encoding()

def terminate_child_processes():
    """终止所有活动子进程"""
    global active_processes
    for proc in active_processes:
        try:
            if proc.poll() is None:  # 检查进程是否仍在运行
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except:
                        pass
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, psutil.NoSuchProcess):
                    pass
        except:
            pass
    active_processes = []

def signal_handler(signum, frame):
    """处理终止信号"""
    print(f"\n接收到终止信号 ({signum})，清理子进程...")
    terminate_child_processes()
    sys.exit(1)

def run_command_realtime(cmd):
    """运行命令并实时输出到终端"""
    global active_processes
    
    try:
        # 创建新进程组配置
        creation_flags = 0
        if platform.system() == "Windows":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Unix系统需要设置新的会话组
            cmd = ["setsid"] + cmd if "setsid" in subprocess.check_output(["which", "setsid"]).decode() else cmd
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding=SYSTEM_ENCODING,
            errors="replace",
            bufsize=1,
            shell=platform.system() == "Windows",
            start_new_session=True,
            creationflags=creation_flags
        )
        
        # 添加到活动进程列表
        active_processes.append(process)
        
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f">>> {output.strip()}")

        stderr = process.stderr.read()
        if stderr:
            print(f"!!! {stderr.strip()}")

        # 从活动进程列表中移除
        if process in active_processes:
            active_processes.remove(process)
            
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

def get_conda_env_path(env_name):
    """获取conda环境的完整路径"""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )
        
        if result.returncode != 0:
            print(f"❌ 获取环境列表失败: {result.stderr}")
            return None
        
        try:
            import json
            envs_data = json.loads(result.stdout)
            for env in envs_data["envs"]:
                # 提取环境名称
                env_base_name = os.path.basename(env)
                if env_base_name == env_name:
                    return env
        except Exception as e:
            print(f"❌ 解析环境列表失败: {str(e)}")
            return None
            
        print(f"❌ 找不到环境: {env_name}")
        return None
        
    except Exception as e:
        print(f"❌ 获取环境路径失败: {str(e)}")
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

def create_environment(requirements_file, env_name: str, python_version: str):
    """根据指定的requirements.txt创建新环境"""
    try:
        req_file = Path(requirements_file)
        if not req_file.exists():
            print(f"❌ 错误: {req_file} 文件不存在")
            return False

        env_name = input(f"请输入新环境名称(默认{DEFAULT_ENV_NAME}): ").strip() if env_name is None else env_name
        if not env_name:
            print("采用默认环境名称: ", DEFAULT_ENV_NAME)
            env_name = DEFAULT_ENV_NAME

        python_version = input("请输入Python版本 (例如 3.11.8): ").strip() if python_version is None else python_version
        if not python_version:
            print("采用默认Python版本: ", DEFAULT_PYTHON_VERSION)
            python_version = DEFAULT_PYTHON_VERSION
        elif not re.match(r"\d+\.\d+\.\d+", python_version):
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



def update_environment(requirements_file, env_name: str = None):
    """使用指定的requirements.txt更新指定环境"""
    try:
        req_file = Path(requirements_file)
        if not req_file.exists():
            print(f"❌ 错误: {req_file} 文件不存在")
            return False

        # 如果没有指定环境名称，使用当前环境
        if env_name is None:
            env_name = input(f"请输入更新环境名称(默认{DEFAULT_ENV_NAME}): ").strip() if env_name is None else env_name
            if not env_name:
                env_name = DEFAULT_ENV_NAME
            print(f"🔄 更新默认环境 '{env_name}'...")
        # 检查指定环境是否存在
        env_path = get_conda_env_path(env_name)
        if not env_path:
            print(f"❌ 环境 '{env_name}' 不存在")
            return False
        print(f"🔄 正在更新环境 '{env_name}'...")


        # 获取目标环境的pip路径
        env_path = get_conda_env_path(env_name)
        if not env_path:
            print(f"❌ 无法获取环境 '{env_name}' 的路径")
            return False
            
        # 打印环境路径用于调试
        print(f"🔍 环境路径: {env_path}")

        pip_exec = "pip.exe" if platform.system() == "Windows" else "pip"
        pip_path = os.path.join(env_path, "bin", pip_exec) if platform.system() != "Windows" else os.path.join(env_path, "Scripts", pip_exec)
        
        if not os.path.exists(pip_path):
            # 尝试使用conda run作为备选方案
            print(f"⚠️ 找不到pip可执行文件: {pip_path}")
            print("🔄 尝试使用conda run执行命令...")
            return_code = run_command_realtime(
                ["conda", "run", "-n", env_name, "pip", "install", "--upgrade", "-r", str(req_file)]
            )
        else:
            # 使用目标环境的pip进行安装
            return_code = run_command_realtime(
                [pip_path, "install", "--upgrade", "-r", str(req_file)]
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
        import traceback
        traceback.print_exc()
        return False

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 主帮助信息
    parser = argparse.ArgumentParser(
        description="Python环境管理工具 - 简化Conda环境创建、导出和更新",
        epilog="使用示例:\n"
               "  导出环境: env_utils.py export (-r export_req.txt)\n"
               "  创建环境: env_utils.py create (-r create_req.txt -e env_name -p python_version)\n"
               "  更新环境: env_utils.py update (-r update_req.txt -e env_name)\n"
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

    # 创建命令
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

    # 更新命令 - 添加环境名称参数
    update_parser = subparsers.add_parser(
        "update", 
        help="更新指定环境",
        description="使用requirements.txt更新指定环境"
    )
    update_parser.add_argument(
        "-r", "--requirements", 
        default=DEFAULT_PIP_FILE,
        metavar="FILE",
        help=f"指定requirements.txt文件路径 (默认: {DEFAULT_PIP_FILE})"
    )
    update_parser.add_argument(
        "-e", "--env-name", 
        default=None,
        metavar="NAME",
        help="指定要更新的环境名称 (默认: 当前激活的环境)"
    )
    update_parser.epilog = "示例:\n" \
                           "  更新当前环境: env_utils.py update -r updated_requirements.txt\n" \
                           "  更新指定环境: env_utils.py update -r updated_requirements.txt -e myenv"

    # 如果没有提供任何参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"🚀 执行命令: {args.command.upper()}")
    print("=" * 60)

    try:
        if args.command == "export":
            success = export_environment(args.output)
        elif args.command == "create":
            success = create_environment(args.requirements, args.env_name, args.python_version)
        elif args.command == "update":
            success = update_environment(args.requirements, args.env_name)
        else:
            print(f"❌ 未知命令: {args.command}")
            sys.exit(1)
    finally:
        # 确保清理所有子进程
        terminate_child_processes()

    print("\n" + "=" * 60)
    print(f"{'✅ 操作成功' if success else '❌ 操作失败'}")
    print("=" * 60)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Python环境管理工具")
    print("=" * 60)
    main()
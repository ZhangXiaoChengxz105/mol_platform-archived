import argparse
import subprocess
import sys
import os
import platform
import re
import datetime
from pathlib import Path

# 全局配置
PIP_FILE = "requirements.txt"
INSTALL_SCRIPT = "install_environment.sh"

def get_system_encoding():
    """获取系统默认编码"""
    try:
        # 简化编码检测
        if platform.system() == "Windows":
            return "utf-8"  # Windows通常使用utf-8
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

        # 实时输出处理
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f">>> {output.strip()}")

        # 检查错误
        stderr = process.stderr.read()
        if stderr:
            print(f"!!! {stderr.strip()}")

        return process.returncode

    except Exception as e:
        print(f"❌ 执行命令失败: {str(e)}")
        return -1

def get_current_env_name():
    """获取当前激活的环境名称"""
    # 方法1: 检查标准环境变量
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if env_name:
        return env_name.split(os.sep)[-1]  # 只取环境名部分

    # 方法2: 使用conda命令查询
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

    # 方法3: 使用pip查看
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

def export_environment():
    """导出当前环境的pip依赖"""
    try:
        env_name = get_current_env_name()
        if not env_name:
            print("❌ 无法确定当前激活的环境")
            print("💡 请确保在Conda环境中运行此命令")
            return False

        print(f"📤 正在导出环境: {env_name}")

        # 获取Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # 获取显式安装的包（用户直接安装的包）
        result = subprocess.run(
            ["pip", "list", "--not-required", "--format=freeze"],
            capture_output=True,
            text=True,
            encoding=SYSTEM_ENCODING,
        )

        if result.returncode != 0:
            print(f"❌ 获取安装包失败: {result.stderr}")
            return False

        # 过滤出用户安装的包
        user_packages = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.startswith(("-e", "@", "#")):
                # 移除平台限制 (如: ; sys_platform == 'win32')
                if ";" in line:
                    line = line.split(";")[0].strip()
                user_packages.append(line)

        # 写入requirements.txt
        with open(PIP_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(user_packages))

        print(f"✅ Pip依赖已保存到: {PIP_FILE}")

        # 生成安装脚本
        script_name = INSTALL_SCRIPT
        if platform.system() == "Windows":
            script_name = "install_environment.bat"

        with open(script_name, "w", encoding="utf-8") as f:
            if platform.system() == "Windows":
                f.write(f"@echo off\n")
                f.write(f":: 自动生成的环境安装脚本 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
                f.write(f"conda create -n {env_name} python={python_version} -y\n")
                f.write(f"call conda activate {env_name}\n")
                f.write(f"pip install -r {PIP_FILE}\n")
                f.write(f"echo 环境安装完成! 使用以下命令激活: conda activate {env_name}\n")
            else:
                f.write("#!/bin/bash\n")
                f.write(f"# 自动生成的环境安装脚本 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
                f.write(f"conda create -n {env_name} python={python_version} -y\n")
                f.write(f"conda activate {env_name}\n")
                f.write(f"pip install -r {PIP_FILE}\n")
                f.write(f"echo \"环境安装完成! 使用以下命令激活: conda activate {env_name}\"\n")
        
        # 设置执行权限 (Linux/macOS)
        if platform.system() != "Windows":
            os.chmod(script_name, 0o755)

        print(f"✅ 安装脚本已生成: {script_name}")
        print("\n💡 在新环境中使用以下命令安装:")
        print(f"   {'双击运行' if platform.system() == 'Windows' else 'bash'} {script_name}")

        return True

    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_environment():
    """根据requirements.txt创建新环境"""
    try:
        # 获取环境名称
        env_name = input("请输入新环境名称: (例如平台环境名称 molplat)").strip()
        if not env_name:
            print("❌ 环境名称不能为空")
            return False

        # 获取Python版本
        python_version = input("请输入Python版本 (例如平台python版本 3.11.8): ").strip()
        if not re.match(r"\d+\.\d+\.\d+", python_version):
            print("❌ 无效的Python版本格式")
            return False

        # 检查requirements.txt是否存在
        if not Path(PIP_FILE).exists():
            print(f"❌ 错误: {PIP_FILE} 文件不存在")
            return False

        # 创建环境
        print(f"🛠️ 正在创建环境 '{env_name}'...")
        print("=" * 80)

        return_code = run_command_realtime(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"]
        )

        if return_code != 0:
            print(f"\n❌ 环境创建失败 (返回码: {return_code})")
            return False

        # 获取环境路径
        env_path = get_conda_env_path(env_name)
        if not env_path:
            print("\n❌ 无法找到环境路径")
            return False

        # 确定pip可执行文件路径
        pip_exec = "pip.exe" if platform.system() == "Windows" else "pip"
        pip_path = os.path.join(env_path, "bin", pip_exec) if platform.system() != "Windows" else os.path.join(env_path, "Scripts", pip_exec)
        
        if not os.path.exists(pip_path):
            print(f"\n❌ 找不到pip可执行文件: {pip_path}")
            return False

        # 安装依赖
        print(f"📦 正在安装依赖...")
        print("=" * 80)

        return_code = run_command_realtime(
            [pip_path, "install", "-r", PIP_FILE]
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
            
        # 解析环境列表输出
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

def update_environment():
    """更新当前环境的依赖"""
    try:
        env_name = get_current_env_name()
        if not env_name:
            print("❌ 无法确定当前激活的环境")
            return False

        print(f"🔄 正在更新环境 '{env_name}'...")

        # 更新依赖
        return_code = run_command_realtime(
            ["pip", "install", "--upgrade", "-r", PIP_FILE]
        )

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
    parser = argparse.ArgumentParser(
        description="Python环境管理工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 导出命令
    subparsers.add_parser("export", help="导出当前环境配置")

    # 创建命令
    subparsers.add_parser("create", help="创建新环境")

    # 更新命令
    subparsers.add_parser("update", help="更新当前环境")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"🚀 执行命令: {args.command.upper()}")
    print("=" * 60)

    if args.command == "export":
        success = export_environment()
    elif args.command == "create":
        success = create_environment()
    elif args.command == "update":
        success = update_environment()
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
import subprocess
import yaml
import argparse
import sys
import os
import platform
import re
from datetime import datetime
from pathlib import Path

# 全局配置
ENVIRONMENT_FILE = "environment.yaml"
PIP_FILE = "requirements.txt"

def read_yaml_with_utf8(file_path):
    """以UTF-8编码读取YAML文件，处理可能的BOM头"""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {str(e)}")
        return None

def create_environment():
    """根据environment.yml创建新环境"""
    try:
        if not Path(ENVIRONMENT_FILE).exists():
            print(f"❌ 错误: {ENVIRONMENT_FILE} 文件不存在")
            return False
        
        env_data = read_yaml_with_utf8(ENVIRONMENT_FILE)
        if env_data is None:
            return False
            
        env_name = env_data.get('name', '')
        if not env_name:
            print("❌ 无法确定环境名称")
            return False
        
        print(f"🛠️ 正在创建环境 '{env_name}'...")
        
        result = subprocess.run(
            ["conda", "env", "create", "--file", ENVIRONMENT_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print(f"✅ 环境 '{env_name}' 创建成功!")
            print(f"👉 使用以下命令激活环境: conda activate {env_name}")
            print("="*60)
            print(result.stdout)
            return True
        else:
            print(f"❌ 创建失败:\n{result.stderr}")
            print("="*60)
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"⚠️ 发生错误: {str(e)}")
        return False

def update_environment():
    """根据environment.yml更新现有环境"""
    try:
        if not Path(ENVIRONMENT_FILE).exists():
            print(f"❌ 错误: {ENVIRONMENT_FILE} 文件不存在")
            return False
        
        env_data = read_yaml_with_utf8(ENVIRONMENT_FILE)
        if env_data is None:
            return False
            
        env_name = env_data.get('name', '')
        if not env_name:
            print("❌ 无法确定环境名称")
            return False
        
        print(f"🔄 正在更新环境 '{env_name}'...")
        
        result = subprocess.run(
            ["conda", "env", "update", "--name", env_name, "--file", ENVIRONMENT_FILE, "--prune"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("✅ 环境更新成功!")
            print("="*60)
            print(result.stdout)
            return True
        else:
            print(f"❌ 更新失败:\n{result.stderr}")
            print("="*60)
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"⚠️ 发生错误: {str(e)}")
        return False

def get_current_env_name():
    """获取当前激活的环境名称（增强版本）"""
    try:
        # 方法1: 使用CONDA_DEFAULT_ENV环境变量（最可靠）
        default_env = os.environ.get("CONDA_DEFAULT_ENV")
        if default_env:
            return default_env
        
        # 方法2: 检查CONDA_PREFIX环境变量
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            # 环境名称通常是路径的最后一部分
            return os.path.basename(conda_prefix)
        
        # 方法3: 使用conda info命令（备选方案）
        try:
            env_info = subprocess.check_output(
                "conda info --envs", 
                shell=True, 
                text=True,
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            )
            for line in env_info.splitlines():
                if line.startswith('*'):
                    # 提取环境名称（星号后的第一个单词）
                    parts = line.split()
                    if len(parts) > 1:
                        return parts[1] if parts[0] == '*' else parts[0]
        except:
            pass
        
        return None
        
    except Exception as e:
        print(f"⚠️ 获取环境名称时出错: {str(e)}")
        return None

def export_environment(platform_independent=True, include_pip=False):
    """导出兼容性环境配置"""
    try:
        env_name = get_current_env_name()
        if not env_name:
            print("❌ 无法确定当前激活的环境")
            print("💡 请确保: ")
            print("1. 你已激活Conda环境")
            print("2. 在正确的终端运行此脚本（如Anaconda Prompt）")
            print("3. Conda已正确安装并添加到系统路径")
            return False
        
        print(f"📤 正在导出环境: {env_name}")
        
        # 1. 获取基础环境信息
        if platform_independent:
            # 跨平台导出：只包含包名和版本，不包含构建号
            print("🔧 使用跨平台兼容模式导出...")
            result = subprocess.run(
                ["conda", "list", "--export"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                print(f"❌ 获取conda包列表失败: {result.stderr}")
                return False
                
            # 处理包列表
            packages = []
            for line in result.stdout.splitlines():
                if "=" in line and not line.startswith("#"):
                    # 去除构建号和平台信息
                    parts = line.split("=")
                    if len(parts) >= 2:
                        # 只保留包名和版本号
                        pkg_entry = f"{parts[0]}={parts[1]}"
                        # 如果包名中包含平台信息(如::win-64)，则去除
                        if "::" in pkg_entry:
                            pkg_entry = pkg_entry.split("::")[-1]
                        packages.append(pkg_entry)
            
            env_config = {
                "name": env_name,
                "channels": ["conda-forge", "defaults"],
                "dependencies": packages
            }
        else:
            # 原始导出方式（包含平台信息）
            print("🔧 使用完整模式导出（包含平台信息）...")
            result = subprocess.run(
                ["conda", "env", "export"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                print(f"❌ 导出环境失败: {result.stderr}")
                return False
                
            env_config = yaml.safe_load(result.stdout)
        
        # 2. 添加元数据
        env_config["metadata"] = {
            "exported": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "platform_independent": platform_independent,
            "pip_included": include_pip
        }
        
        # 3. 处理pip依赖
        if include_pip:
            print("🔍 收集pip安装的包...")
            pip_result = subprocess.run(
                ["pip", "freeze"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            if pip_result.returncode != 0:
                print(f"❌ 获取pip包列表失败: {pip_result.stderr}")
                return False
                
            pip_packages = pip_result.stdout.splitlines()
            
            # 过滤掉非标准包（如可编辑安装或路径依赖）
            clean_pip_packages = []
            for pkg in pip_packages:
                # 跳过可编辑安装和路径依赖
                if pkg.startswith("-e ") or "@ file" in pkg:
                    print(f"⚠️ 跳过特殊依赖: {pkg}")
                    continue
                # 只保留包名和版本
                if "==" in pkg:
                    clean_pip_packages.append(pkg.split("==")[0] + "==" + pkg.split("==")[1])
                else:
                    clean_pip_packages.append(pkg)
            
            # 创建独立的pip配置节
            pip_section = {"pip": clean_pip_packages}
            env_config["dependencies"].append(pip_section)
            
            # 单独保存pip依赖
            with open(PIP_FILE, "w", encoding='utf-8') as f:
                f.write("\n".join(clean_pip_packages))
            print(f"💾 Pip依赖已保存到: {PIP_FILE}")
        
        # 4. 保存环境文件
        with open(ENVIRONMENT_FILE, "w", encoding='utf-8') as f:
            yaml.dump(env_config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 环境配置已保存到: {ENVIRONMENT_FILE}")
        print("="*60)
        print("💡 新用户安装指南:")
        print(f"1. 创建环境: conda env create -f {ENVIRONMENT_FILE}")
        print(f"2. 激活环境: conda activate {env_name}")
        if include_pip:
            print(f"3. (可选)安装pip依赖: pip install -r {PIP_FILE}")
        print("="*60)
        
        # 打印导出内容预览
        print("📄 导出文件预览 (前20行):")
        with open(ENVIRONMENT_FILE, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 20:
                    print(line.rstrip())
                else:
                    print("...")
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="环境管理工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建命令
    create_parser = subparsers.add_parser('create', help='创建新环境')
    
    # 更新命令
    update_parser = subparsers.add_parser('update', help='更新当前环境')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出当前环境配置')
    export_parser.add_argument('--full', action='store_true', 
                              help='导出完整环境（包含平台特定信息）')
    export_parser.add_argument('--pip', action='store_true', 
                              help='导出pip依赖')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'create':
            success = create_environment()
        elif args.command == 'update':
            success = update_environment()
        elif args.command == 'export':
            success = export_environment(
                platform_independent=not args.full, 
                include_pip=args.pip
            )
        else:
            print(f"未知命令: {args.command}")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)

if __name__ == "__main__":
    print("="*60)
    print("Conda环境管理工具")
    print("="*60)
    main()
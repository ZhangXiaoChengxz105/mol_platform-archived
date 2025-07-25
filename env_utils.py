import subprocess
import yaml
import argparse
import sys
import os
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
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print(f"✅ 环境 '{env_name}' 创建成功!")
            print(f"👉 使用以下命令激活环境: conda activate {env_name}")
            return True
        else:
            print(f"❌ 创建失败:\n{result.stderr}")
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
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("✅ 环境更新成功!")
            return True
        else:
            print(f"❌ 更新失败:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️ 发生错误: {str(e)}")
        return False

def export_environment(minimal=True, include_pip=False):
    """导出当前环境配置到标准文件"""
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
        
        cmd = "conda env export --from-history" if minimal else "conda env export"
        env_data = subprocess.check_output(
            cmd, 
            shell=True, 
            text=True,
            encoding='utf-8'
        )
        
        env_config = yaml.safe_load(env_data)
        
        env_config["metadata"] = {
            "exported": datetime.now().isoformat(),
            "minimal": minimal,
            "pip_included": include_pip
        }
        
        if include_pip:
            print("🔍 收集pip安装的包...")
            pip_packages = subprocess.check_output(
                ["pip", "freeze"], 
                text=True,
                encoding='utf-8'
            ).splitlines()
            
            pip_section = next((item for item in env_config.get("dependencies", []) 
                               if isinstance(item, dict) and "pip" in item), None)
            
            if not pip_section:
                pip_section = {"pip": []}
                env_config.setdefault("dependencies", []).append(pip_section)
            
            existing_pip = set(pkg.split("==")[0] for pkg in pip_section["pip"])
            for pkg in pip_packages:
                pkg_name = pkg.split("==")[0]
                if pkg_name not in existing_pip:
                    pip_section["pip"].append(pkg)
            
            with open(PIP_FILE, "w", encoding='utf-8') as f:
                f.write("\n".join(pip_packages))
            print(f"💾 Pip依赖已保存到: {PIP_FILE}")
        
        with open(ENVIRONMENT_FILE, "w", encoding='utf-8') as f:
            yaml.dump(env_config, f, sort_keys=False, default_flow_style=False)
        
        print(f"✅ 环境配置已保存到: {ENVIRONMENT_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
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
                stderr=subprocess.DEVNULL  # 忽略错误输出
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

def main():
    parser = argparse.ArgumentParser(description="环境管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    subparsers.add_parser('create', help='创建新环境')
    subparsers.add_parser('update', help='更新当前环境')
    
    parser_export = subparsers.add_parser('export', help='导出当前环境配置')
    parser_export.add_argument('--minimal', action='store_true', help='精简模式')
    parser_export.add_argument('--pip', action='store_true', help='导出pip依赖')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'create':
        success = create_environment()
    elif args.command == 'update':
        success = update_environment()
    elif args.command == 'export':
        success = export_environment(
            minimal=args.minimal, 
            include_pip=args.pip
        )
    else:
        print(f"未知命令: {args.command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
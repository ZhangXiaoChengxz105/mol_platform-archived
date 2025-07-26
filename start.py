import os
import subprocess
import platform

def check_initialization():
    """检查初始化状态并执行相应操作"""
    # 初始化标记文件路径（存储在用户目录下）
    init_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit_init_flag")
    
    # 检查是否是首次运行
    if not os.path.exists(init_flag):
        response = input("检测到首次使用，是否初始化环境？(y/n): ").strip().lower()
        if response == 'y':
            print("开始初始化环境...")
            # 执行初始化操作（这里替换为你的初始化命令）
            init_success = perform_initialization()
            
            if init_success:
                # 创建初始化完成标记
                with open(init_flag, 'w') as f:
                    f.write("initialized")
                print("环境初始化完成！")
            else:
                print("环境初始化失败，请手动检查environment.md")
        else:
            print("跳过初始化，直接启动应用")
    else:
        response = input("是否更新环境？(y/n): ").strip().lower()
        if response == 'y':
            print("开始更新环境...")
            # 执行更新操作（这里替换为你的更新命令）
            update_success = perform_update()
            
            if update_success:
                print("环境更新完成！")
            else:
                print("环境更新失败，请手动检查environment.md")

def perform_initialization():
    """执行初始化操作，返回是否成功"""
    try:
        # 示例：安装依赖
        # 根据系统类型执行不同命令
        subprocess.run(["python", "env_utils.py", "create", "-r", "requirements.txt", "-e", "molplat", "-p", "3.11.8"], check=True)
        
        # 示例：创建必要目录
        os.makedirs("data", exist_ok=True)
        
        # 添加其他初始化任务...
        return True
    except Exception as e:
        print(f"初始化过程中出错: {e}")
        return False

def perform_update():
    """执行更新操作，返回是否成功"""
    try:
        # 示例：更新依赖
        subprocess.run(["python", "env_utils.py", "update", "-r", "requirements.txt"], check=True)
        
        # 添加其他更新任务...
        return True
    except Exception as e:
        print(f"更新过程中出错: {e}")
        return False

def run_streamlit():
    """启动Streamlit应用"""
    streamlit_script = os.path.join("result_analysis", "app.py")

    # 设置环境变量
    env = os.environ.copy()
    env["STREAMLIT_SUPPRESS_EMAIL_LOGGING"] = "true"
    env["BROWSER"] = "default"

    # 在 Windows 上使用 shell=True
    shell = platform.system() == "Windows"

    subprocess.run(["streamlit", "run", streamlit_script], env=env, shell=shell)

if __name__ == "__main__":
    # 检查并执行初始化/更新
    check_initialization()
    
    # 启动主应用
    run_streamlit()
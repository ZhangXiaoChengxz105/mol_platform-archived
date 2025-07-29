import os
import platform
import signal
import sys
from env_utils import create_environment, update_environment, get_current_env_name, get_conda_env_path, run_command_realtime
import subprocess
import socket

INIT_FLAG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit_init_flag")

def check_initialization():
    """检查初始化状态并执行相应操作"""
    # 初始化标记文件路径
    
    # 检查是否是首次运行
    if not os.path.exists(INIT_FLAG_PATH):
        response = ""
        while response not in ["y","yes","n","no"]:
            response = input("检测到首次使用，是否初始化平台运行环境？(y/n): ").strip().lower()
            if response not in ["y","yes","n","no"]:
                print("请输入: y/yes or n/no")

        if response in ["y","yes"]:
            print("\n开始初始化环境...")
            # 执行初始化操作
            perform_initialization()

        else:
            print("跳过初始化，直接启动应用")
    else:
        with open (INIT_FLAG_PATH) as f:
            base_env = f.read()
        print(f"已初始化平台，初始化平台运行环境为: {base_env}")
        cur_env = get_current_env_name()
        
        response = input(f"是否更新环境？（平台默认环境molplat，当前环境{cur_env}，可指定更新环境, 默认不更新）(y/n): ").strip().lower()
        if response in ["y","yes"]:
            print(f"开始更新环境...")
            # 执行更新操作
            update_success = perform_update()
            
            if update_success:
                print("环境更新完成！")
            else:
                print("环境更新失败，请手动检查environment.md")
        else:
            print("跳过更新")
def perform_initialization():
    """执行初始化操作，返回是否成功"""
    try:
        env_name = input("请输入平台初始环境名称(默认molplat): ").strip()
        if not env_name:
            env_name = "molplat"
            print("采用默认环境名称: ", "molplat")
        # 直接调用env_utils中的函数创建环境
        success = create_environment(
            base_requirements="requirements.txt",
            env_name=env_name,
            python_version="3.11.8"
        )
        if success:
            print("环境创建完成！\n")
            print("生成平台环境管理文件environment.yaml")
            # config = {env_name: {"molplat": "requirements.txt"}}
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'environment.yaml')
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(f"{env_name}:\n  molplat: requirements.txt\n")

                # 创建初始化完成标记
            with open(INIT_FLAG_PATH, 'w') as f:
                f.write(env_name)
            print("环境初始化完成！")
        return success
    except Exception as e:
        print(f"初始化过程中出错: {e}")
        print("环境初始化失败，请手动检查environment.md")
        exit(1)
        return False

def perform_update():
    """执行更新操作，返回是否成功"""
    try:
        # 直接调用env_utils中的函数更新环境
        return update_environment(requirements_file="requirements.txt")
    except Exception as e:
        print(f"更新过程中出错: {e}")
        return False

def get_local_ip():
    try:
        # 使用 UDP socket 连接外部 IP，不实际发送数据
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS，仅用于获取本机出口 IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "127.0.0.1" 

def run_streamlit(env_name):
    """启动Streamlit应用并返回进程对象"""
    streamlit_script = os.path.join("result_analysis", "app.py")
    
    # 设置环境变量
    env = os.environ.copy()
    env["STREAMLIT_SUPPRESS_EMAIL_LOGGING"] = "true"
    env["BROWSER"] = "default"

    # 检查环境是否存在
    env_path = get_conda_env_path(env_name)
    if not env_path:
        print(f"❌ 环境 '{env_name}' 不存在！")
        print("请指定正确的环境名称(使用初始化创建的环境名)")
        return None
    response = ""
    while response not in ["y", "yes", "n", "no"]:
        response = input("是否启用服务器版本，使局域网内部设备能够访问此应用，默认为是: ").strip().lower()
        if response not in ["y", "yes", "n", "no"]:
            response = "yes"  # 默认 yes

    if response in ['no', 'n']:
        print(f"🚀 在环境 '{env_name}' 中启动应用...(不启动服务器，仅限本机使用)")
        print(f"📜 启动streamlit应用: {streamlit_script}")
        cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "streamlit", "run", streamlit_script]
    else:
        print(f"🚀 在环境 '{env_name}' 中启动应用...(启动服务器，局域网内设备均可访问)")
        print(f"📜 启动streamlit应用: {streamlit_script}")
        ip = get_local_ip()
        print(f"📜 服务器部署在地址: {ip}, 服务器所在端口请查看接下来的输出")
        cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "streamlit", "run", streamlit_script, "--server.address=0.0.0.0",'--browser.serverAddress=localhost']

    # 启动进程并返回引用
    return subprocess.Popen(
        cmd,
        env=env,
        
    )


if __name__ == "__main__":
    # 检查并执行初始化/更新
    check_initialization()
    
    env_name = input("\n指定初始平台运行环境（默认molplat，不包含模型配置）: ").strip().lower()
    env_name = env_name if env_name else "molplat"

    # 启动主应用
    streamlit_proc = run_streamlit(env_name)

import os
import platform
import signal
import sys
from env_utils import create_environment, update_environment, get_current_env_name
import subprocess

def check_initialization():
    """检查初始化状态并执行相应操作"""
    # 初始化标记文件路径
    init_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit_init_flag")
    
    # 检查是否是首次运行
    if not os.path.exists(init_flag):
        response = input("检测到首次使用，是否初始化环境？(y/n): ").strip().lower()
        if response == 'y':
            print("开始初始化环境...")
            # 执行初始化操作
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
        cur_env = get_current_env_name()
        
        response = input(f"是否更新环境？（平台默认环境molplat，当前环境{cur_env}，可指定更新环境）(y/n): ").strip().lower()
        if response == 'y':
            print(f"开始更新环境...")
            # 执行更新操作
            update_success = perform_update()
            
            if update_success:
                print("环境更新完成！")
            else:
                print("环境更新失败，请手动检查environment.md")

def perform_initialization():
    """执行初始化操作，返回是否成功"""
    try:
        # 直接调用env_utils中的函数创建环境
        success = create_environment(
            requirements_file="requirements.txt",
            env_name="test",
            python_version="3.11.8"
        )
        
        # 创建数据目录
        os.makedirs("data", exist_ok=True)
        return success
    except Exception as e:
        print(f"初始化过程中出错: {e}")
        return False

def perform_update():
    """执行更新操作，返回是否成功"""
    try:
        # 直接调用env_utils中的函数更新环境
        return update_environment(requirements_file="requirements.txt")
    except Exception as e:
        print(f"更新过程中出错: {e}")
        return False

def run_streamlit():
    """启动Streamlit应用并返回进程对象"""
    streamlit_script = os.path.join("result_analysis", "app.py")
    
    # 设置环境变量
    env = os.environ.copy()
    env["STREAMLIT_SUPPRESS_EMAIL_LOGGING"] = "true"
    env["BROWSER"] = "default"

    
    # 启动进程并返回引用
    return subprocess.run(
        ["streamlit", "run", streamlit_script],
        env=env,
        start_new_session=True,  # 创建新的进程组
    )

def terminate_process(proc):
    """跨平台终止进程及其子进程"""
    try:
        if platform.system() == "Windows":
            # Windows系统发送CTRL_BREAK信号
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # Unix系统终止整个进程组
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        
        # 等待进程结束
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            # 如果超时，强制终止
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass

if __name__ == "__main__":
    # 检查并执行初始化/更新
    check_initialization()
    
    # 启动主应用
    streamlit_proc = run_streamlit()
    
    # 注册信号处理
    def handle_exit(signum, frame):
        print("\n终止Streamlit服务...")
        terminate_process(streamlit_proc)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # 等待进程结束
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        handle_exit(signal.SIGINT, None)
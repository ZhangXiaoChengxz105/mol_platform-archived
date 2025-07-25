import os
import subprocess
import platform

def run_streamlit():
    streamlit_script = os.path.join("result_analysis", "app.py")

    # 设置环境变量
    env = os.environ.copy()
    env["STREAMLIT_SUPPRESS_EMAIL_LOGGING"] = "true"       # 跳过邮箱提示
    env["BROWSER"] = "default"                              # 强制默认浏览器打开

    # 在 Windows 上使用 shell=True 以支持 PATH 中的可执行程序（如 streamlit）
    shell = platform.system() == "Windows"

    subprocess.run(["streamlit", "run", streamlit_script], env=env, shell=shell)

if __name__ == "__main__":
    run_streamlit()
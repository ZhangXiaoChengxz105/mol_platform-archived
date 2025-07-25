import subprocess

def run_streamlit():
    streamlit_script = "result_analysis/app.py"
    # 直接运行 streamlit run，假设当前环境已激活
    subprocess.run(["streamlit", "run", streamlit_script])

if __name__ == "__main__":
    run_streamlit()
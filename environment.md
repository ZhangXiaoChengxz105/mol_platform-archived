# 使用脚本：
## env_utils.py
使用示例:

    导出环境: env_utils.py export (-r export_req.txt -e env_name -p python_version)\n
    创建环境: env_utils.py create (-r create_req.txt)\n
    更新环境: env_utils.py update (-r update_req.txt)\n
    默认路径: -r requirements.txt
    默认环境名：-e molplat
    默认Python版本: -p 3.11.8

## install_molplat.bat (windows script)
    根据requirements.txt初始化平台环境
# 自主安装
## 创建并激活新环境
    conda create -n molplat python=3.11.8 -y
    conda activate molplat
## 安装依赖
    pip install pandas pyyaml matplotlib seaborn scikit-learn torch numpy streamlit streamlit-option-menu
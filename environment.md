# 开发者环境工具：
## env_utils.py
使用示例:
    查看帮助: env_utils.py -h or help
    导出环境: env_utils.py export (-r export_req.txt)
    创建环境: env_utils.py create (-r create_req.txt -a additional_req.txt ... -e env_name -p python_version)
    更新环境: env_utils.py update (-r update_req.txt -a additional_req.txt ... -e env_name)
    默认路径: -r requirements.txt
    默认环境名：-e molplat
    默认Python版本: -p 3.11.8

# 自主安装平台初始环境
## 创建并激活新环境
    conda create -n molplat python=3.11.8 -y
    conda activate molplat
## 安装依赖
    pip install pandas pyyaml matplotlib seaborn scikit-learn torch numpy streamlit streamlit-option-menu
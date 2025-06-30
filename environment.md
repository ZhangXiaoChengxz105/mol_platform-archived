# 创建并激活新环境
conda create -n molplat python=3.11.8 -y
conda activate molplat
# 安装依赖
pip install torch torch-geometric rdkit transformers pandas
pip install seaborn matplotlib pyyaml numpy scikit-learn
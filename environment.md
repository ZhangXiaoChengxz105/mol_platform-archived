# 创建并激活新环境
conda create -n molplat python=3.11.8 -y
conda activate molplat
# 安装依赖
pip install torch torch-geometric rdkit==2024.3.5 transformers pandas

pip install seaborn matplotlib pyyaml numpy scikit-learn==1.7.0 xgboost streamlit-option-menu streamlit

## 克隆并安装fast-transformers
conda install -c conda-forge cxx-compiler -y
git clone https://github.com/idiap/fast-transformers.git
### （其他克隆方法）
git clone https://hub.yzuu.cf/idiap/fast-transformers.git
git clone git@github.com:idiap/fast-transformers.git
git clone https://gitclone.com/github.com/idiap/fast-transformers.git
## 安装
cd fast-transformers
pip install -e .    # 注意，windows需要MSVC，运行cl.exe以检查
cd ../
## 路径设置
set LD_LIBRARY_PATH=%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH% # windows
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # linux

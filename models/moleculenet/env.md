# **环境配置**
    python=3.11.8
    
    pip install torch torch-geometric rdkit==2024.3.5 numpy scikit-learn==1.7.0 transformers pandas xgboost
可根据requirements.txt(平台基础依赖), 使用env_utils.py，快速创建独立模型环境，安装依赖

    python env_utils.py create -r requirements.txt models/moleculenet/ -e molplat -p 3.11.8
环境创建后请在使用平台时指定使用模型工作流对应的环境
    
## **seq 额外配置**
seq 工作流需使用编译版本的 fast-transformers：
    
## 配置fast-transformers：
    conda install -c conda-forge cxx-compiler -y
    git clone https://github.com/idiap/fast-transformers.git
### （其他克隆方法）
    git clone https://hub.yzuu.cf/idiap/fast-transformers.git
    git clone git@github.com:idiap/fast-transformers.git
    git clone https://gitclone.com/github.com/idiap/fast-transformers.git
### 安装
    cd fast-transformers
    pip install -e .    # 注意，windows需要MSVC，运行cl.exe以检查
    cd ../
### 路径设置
    set LD_LIBRARY_PATH=%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH% # windows
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # linux
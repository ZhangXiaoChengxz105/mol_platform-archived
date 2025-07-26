@echo off
:: 自动生成的环境安装脚本 (2025-07-26 13:37)
conda create -n test python=3.11.8 -y
call conda activate test
pip install -r requirements.txt
echo 环境安装完成! 使用以下命令激活: conda activate test

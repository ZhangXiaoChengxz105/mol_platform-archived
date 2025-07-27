# **mol_platform**
**本平台强制依赖 Conda 环境管理系统**，请确保已安装 Conda 后再进行操作。

▶ 平台启动方式（默认当前目录为平台根目录）

    python start.py
（包含环境初始化功能）
或

    streamlit run result_analysis/app.py
（需自行安装平台所需环境，参考environment.md）

▶ 模型上传与使用
平台提供模型文件上传功能，需单独配置模型运行环境（参考配置说明env.md或配置文件requirements.txt）
可使用平台提供的环境管理工具：
运行

    python env_utils.py -h

以查看环境配置工具的使用说明

▶ 平台刷新
平台刷新按钮 - rerun，开发调试常用
由于平台代码中包含对本地文件、环境的访问读写操作，有部分操作可能导致平台提示File Change，此时需要用户手动刷新更新状态，或退出重启平台

▶ 平台关闭
平台关闭按钮 - “关闭”，或终端关闭进程

▶ 提供模版模型文件moleculenet_model.zip， 数据文件moleculenet_data.zip
获取链接：

    https://zjuintl-my.sharepoint.com/:f:/g/personal/yanzhen_22_intl_zju_edu_cn/Egl_NYBwizhAkf_Ia0VZWyIBf2Vz8NN5mDLX7EDPPhKlRA
有效期至： 2025/10/25



浙江大学数据科学研究中心
Miao Lab@ZheJiang University

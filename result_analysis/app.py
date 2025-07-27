import streamlit as st
import yaml
import os
import sys
import subprocess
import pathlib
import pandas as pd
import re
import json
from datetime import datetime
from process import process, delete
try:
    project_root = pathlib.Path(__file__).resolve().parents[1]
except NameError:
    project_root = pathlib.Path(os.getcwd()).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from models.check_utils import get_datasets_measure_names,CheckUtils
from streamlit_option_menu import option_menu


def render_scrollable_markdown(md_text, height=300):
    st.markdown(
        f"""
        <div style='height:{height}px; overflow:auto; padding:10px; border:1px solid #ccc; background-color:#f9f9f9; border-radius:5px'>
        {md_text}
        </div>
        """,
        unsafe_allow_html=True
    )
    
def set_streamlit_upload_limit(limit_mb=2048):
    config_dir = os.path.expanduser("~/.streamlit")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")

    with open(config_path, "w") as f:
        f.write(f"[server]\nmaxUploadSize = {limit_mb}\n")

set_streamlit_upload_limit(2048)

st.set_page_config(layout="wide")
st.title("分子性质预测集成平台")
st.markdown("集模型和数据管理于一体，支持上传删减模型，一键式选择，处理数据，并且展示")

# ----------- 配置路径 -----------
MODEL_PATH =os.path.join(project_root,'models')
CONFIG_PATH = os.path.join(project_root,'result_analysis','config_run.yaml')
# MODEL_MAP_PATH = os.path.join(project_root,'models','model_datasets.yaml')
RUN_SCRIPT_PATH = os.path.join(project_root,'result_analysis','run_all.py')
HISTORY_PATH = os.path.join(project_root, 'results', 'results','run_history.json')
MODEL_DATASET_PATH = os.path.join(MODEL_PATH,'models.yaml')
UPLOAD_MODEL_README = os.path.join(MODEL_PATH,'models_README.md')
UPLOAD_DATA_README = os.path.join(project_root,'dataset','dataset_README.md')





# ----------- 加载 config.yaml -----------
@st.cache_data
def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        return {
            "model": "fp",
            "name": "BBBP",
            "eval": True,
            "target_list": "all",
            "smiles_list": "random200",
            "output": "results",
            "plotpath": "plots",
            "plotprevisousruns": False
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_all_model_types():
    with open(MODEL_DATASET_PATH,'r') as f:
        config = yaml.safe_load(f)
        return list(config.keys())

def get_models_and_data(top_key):  # top_key 是 'moleculenet'
    with open(MODEL_DATASET_PATH, 'r') as f:
        config = yaml.safe_load(f)

    top_config = config.get(top_key, {})
    # 提取所有模型名组合，如 FP_NN, GNN_GIN 等
    models_config = top_config.get('models', {})
    model_names = []
    for model_type in models_config:
        if isinstance(models_config[model_type], dict): 
            model_names.append(model_type)
    DATACONFIG_PATH = os.path.join(project_root,'dataset','data',top_key,'dataset.yaml')
    with open(DATACONFIG_PATH, 'r',encoding='utf-8') as g:
        config = yaml.safe_load(g)
    all_datasets = config.get('dataset_names',[])

    return model_names, all_datasets
        
def get_data_type(top_key):
    DATACONFIG_PATH = os.path.join(project_root,'dataset','data',top_key,'dataset.yaml')
    with open(DATACONFIG_PATH, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return (config.get('data_type',''))

def display_csv_tables(csv_dir):
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(csv_dir, csv_file)
        with st.expander(f"📄 {csv_file}"):
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"{csv_file} 加载失败: {e}")
                
def display_images_recursively(base_dir):
    for root, dirs, files in os.walk(base_dir):
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if image_files:
            rel_path = os.path.relpath(root, base_dir)
            with st.expander(f"📂 {rel_path}"):
                cols = st.columns(2)  # 每行两列
                for idx, image in enumerate(sorted(image_files)):
                    image_path = os.path.join(root, image)
                    col = cols[idx % 2]  # 交替写入两个列
                    with col:
                        st.image(image_path, caption=image, use_container_width="always")

                
def get_latest_run_folder(base="results"):
    run_dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and re.match(r"run\d+", d)]
    run_numbers = [int(re.findall(r"run(\d+)", d)[0]) for d in run_dirs]
    if run_numbers:
        latest_run = f"run{max(run_numbers)}"
        return latest_run,os.path.join(base, latest_run)
    return None

def get_submodel(model_type, model):
    with open(MODEL_DATASET_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    try:
        return list(data[model_type]['models'][model].keys())
    except (KeyError, AttributeError):
        return []

        


def show_file_selector(label: str, file_path: str, is_markdown: bool = False, is_text: bool = False, height: int = 500) -> None:
    """显示复选框，勾选后展示文件内容，支持 markdown、python 和 txt 形式，带固定高度滚动条"""
    if not os.path.exists(file_path):
        st.write(f"{label} 文件不存在：{file_path}")
        return

    show_content = st.checkbox(f"显示 {label}", key=f"show_{label}")

    if show_content:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if is_markdown:
            render_scrollable_markdown(content, height=height)
        elif is_text:
            st.code(content, language=None, line_numbers=True, height=height)  # txt 内容无高亮
        else:
            st.code(content, language="python", line_numbers=True, height=height)


# ----------- 保存 config.yaml -----------
def save_config(config, path=CONFIG_PATH):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
        
def list_to_csv_fields(config_dict, fields):
    for field in fields:
        if isinstance(config_dict.get(field), list):
            config_dict[field] = ",".join(str(x) for x in config_dict[field])
    return config_dict


def get_datasets_for_model(model_list, model_map):
    """
    从模型列表中提取所有模型支持的数据集，并返回它们的交集。

    参数：
    - model_list (List[str]): 模型名称列表，如 ['FP NN', 'GNN GCN']
    - model_map (Dict[str, Dict]): 从 model_datasets.yaml 加载的模型映射

    返回：
    - List[str]: 所有模型共同支持的数据集名称列表
    """
    all_dataset_sets = []

    for model in model_list:
        try:
            model_name= model.split("_")[0]
            model_type = model.split("_")[1]
        except ValueError:
            continue  # 忽略格式错误的条目

        datasets = model_map.get(model_name, {}).get(model_type)
        if datasets:
            all_dataset_sets.append(set(datasets))

    if not all_dataset_sets:
        return []

    common_datasets = set.intersection(*all_dataset_sets)
    return sorted(list(common_datasets))





# ----------- 初始化 session_state -----------
if "selected_model_field" not in st.session_state:
    st.session_state["selected_model_field"] = None
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = []
if "selected_datasets" not in st.session_state:
    st.session_state["selected_datasets"] = []
if "eval" not in st.session_state:
    st.session_state["eval"] = True
if "smiles_list" not in st.session_state:
    st.session_state["smiles_list"] = "random200"
if "smiles_input_mode" not in st.session_state:
    st.session_state["smiles_input_mode"] = "auto_eval"  # 可选: auto_eval, file_upload, manual_input
if "smiles_text_input" not in st.session_state:
    st.session_state["smiles_text_input"] = ""
if "smiles_file" not in st.session_state:
    st.session_state["smiles_file"] = None
if "smiles_eval_mode" not in st.session_state:
    st.session_state["smiles_eval_mode"] = "random"
if "smiles_eval_num" not in st.session_state:
    st.session_state["smiles_eval_num"] = 200
def get_top_level_keys():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        return list(data.keys())
    else:
        return []
def run_long_command(cmd, description="正在执行命令..."):
    import subprocess
    import time

    with st.spinner(description):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                st.error("命令执行失败 ❌")
                if stderr:
                    st.code(stderr, language="bash")
                return False
            else:
                if stdout:
                    st.code(stdout, language="bash")
                return True
        except Exception as e:
            st.error("执行过程中发生异常 ❌")
            st.exception(e)
            return False

def show_update_button(model, reqname):
    with st.expander("更新环境"):
        keys = get_top_level_keys()
        if not keys:
            st.warning("environment.yaml 文件为空或不存在，无法选择环境名。")
            return

        env_name = st.selectbox("选择环境名字", keys)

        if st.button("Update"):
            st.text("⏳ 开始更新...")
            success = update(reqname, env_name, model)
            if success:
                st.success(f"✅ Update 成功：model={model}, reqname={reqname}, envname={env_name}")
                st.text("请退出并重新打开以生效")
            else:
                st.error("❌ Update 失败，请检查输出信息")

def update(file, envname, model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(current_dir, '../env_utils.py'))
    env_md_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    cmd = [sys.executable, script_path, "update", '-r', file, '-e', envname]
    success = run_long_command(cmd, description=f"正在更新环境 {envname}...")

    if not success:
        return False

    # ✅ 保留你的 environment.yaml 写入逻辑
    try:
        with open(env_md_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        if envname not in data:
            st.error(f"错误: environment.yaml 顶层找不到环境名 '{envname}'")
            return False

        data[envname][model] = file

        with open(env_md_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)

        return True

    except Exception as e:
        st.error(f"写入 environment.yaml 失败: {e}")
        return False

def show_create_button(reqname, model):
    with st.expander("创建环境"):
        st.markdown("### 创建模型配置")
        col1, col2 = st.columns(2)

        with col1:
            py_version = st.text_input("Python 版本", value="3.8", max_chars=10)

        with col2:
            env_name = st.text_input("环境名字", max_chars=20)

        if st.button("Create"):
            if not py_version.strip() or not env_name.strip():
                st.error("请填写完整的 Python 版本和环境名字！")
            else:
                create(model, reqname, env_name, py_version)
                st.text("创建环境中")
                st.success(f"Create 调用成功，环境名={env_name}, Python版本={py_version}")
                st.text("创建新环境，请退出重新打开")

def create(model, file, envname, version):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(current_dir, '../env_utils.py'))
    base_reqs = os.path.abspath(os.path.join(current_dir, '../requirements.txt'))
    env_md_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    cmd = [sys.executable, script_path, 'create', '-r', base_reqs, '-a', file, '-e', envname, '-p', version]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("创建成功，输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"创建失败，返回码：{e.returncode}")
        print(e.stderr)
        return

    with open(env_md_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if envname not in data or not isinstance(data[envname], dict):
        data[envname] = {}

    data[envname][model] = file

    with open(env_md_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True)

def on_select_change():
    # 选框改变时，如果选择“自定义输入”，保持final_model_type不变等待输入框输入
    # 否则更新final_model_type，并标记列表需刷新
    selected = st.session_state["model_type_select"]
    if selected != "自定义输入":
        if st.session_state.get("final_model_type", "") != selected:
            st.session_state["final_model_type"] = selected
            st.session_state["model_list_changed"] = True

def on_custom_input_change():
    # 自定义输入框改变时，更新final_model_type并标记刷新
    text = st.session_state.get("custom_model_input", "").strip()
    if st.session_state.get("final_model_type", "") != text:
        st.session_state["final_model_type"] = text
        st.session_state["model_list_changed"] = True

# 顶部按钮
if "final_model_type" not in st.session_state:
    st.session_state["final_model_type"] = ""
if "uploaded_model_zip" not in st.session_state:
    st.session_state["uploaded_model_zip"] = None
if "uploaded_model_config" not in st.session_state:
    st.session_state["uploaded_model_config"] = None
if "uploaded_data_zip" not in st.session_state:
    st.session_state["uploaded_data_zip"] = None
if "uploaded_data_config" not in st.session_state:
    st.session_state["uploaded_data_config"] = None
if "show_model_input" not in st.session_state:
    st.session_state["show_model_input"] = False
if "model_list_changed" not in st.session_state:
    st.session_state["model_list_changed"] = True

# ----------- 展开按钮 -----------
close_tab_js = """
<script>
    window.close();
</script>
"""
exit_col_space, exit_col_btn = st.columns([9, 1])
with exit_col_btn:
    if st.button("退❌出"):
        st.warning("程序即将关闭...")
        st.components.v1.html(close_tab_js)
        os._exit(0)


col1, col2 = st.columns([10, 2])
with col2:
    if st.button("➕ 添加数据集与模型（再点击一次以返回）"):
        st.session_state["show_model_input"] = not st.session_state["show_model_input"]

# ----------- 展开区域 -----------
if st.session_state.get("show_model_input", True):

    st.markdown("#### 🔧 自定义数据集类型与模型包上传")
    st.markdown("** 注意，如果模型依赖python库，请在终端自行安装以避免冲突")

    # 上传说明文件展示
    if os.path.exists(UPLOAD_MODEL_README):
        with open(UPLOAD_MODEL_README, "r", encoding="utf-8") as f:
            model_readme_text = f.read()
        with st.expander("📘 查看模型上传说明 (MODEL_readme.md)"):
            render_scrollable_markdown(model_readme_text, height=600)

    if os.path.exists(UPLOAD_DATA_README):
        with open(UPLOAD_DATA_README, "r", encoding="utf-8") as f:
            data_readme_text = f.read()
        with st.expander("📗 查看数据上传说明 (DATASET_readme.md)"):
            render_scrollable_markdown(data_readme_text, height=600)

    # 获取所有模型类型
    try:
        all_model_types = get_all_model_types()
    except Exception as e:
        st.warning(f"加载数据集类型失败：{e}")
        all_model_types = []

    model_type_options = ["自定义输入"] + all_model_types

    # 计算当前选中index，默认选自定义输入
    if st.session_state["final_model_type"] in all_model_types:
        current_index = model_type_options.index(st.session_state["final_model_type"])
    else:
        current_index = 0

    selected_option = st.selectbox(
        "从已有数据集类型中选择或直接输入新类型：",
        options=model_type_options,
        index=current_index,
        key="model_type_select",
        on_change=on_select_change,
    )

    if selected_option == "自定义输入":
        custom_input = st.text_input(
            "请输入新的数据集类型并回车",
            value=st.session_state.get("custom_model_input", ""),
            key="custom_model_input",
            on_change=on_custom_input_change,
        )
    else:
        if "custom_model_input" in st.session_state:
            del st.session_state["custom_model_input"]

    if selected_option != "自定义输入" and st.session_state.get("final_model_type"):

        if st.session_state["model_list_changed"]:
            # 只有非自定义输入，且列表改变时，加载列表
            models_list, datasets_list = get_models_and_data(st.session_state["final_model_type"])
            st.session_state["models_list"] = models_list
            st.session_state["datasets_list"] = datasets_list
            st.session_state["model_list_changed"] = False

        datatype = get_data_type(st.session_state["final_model_type"])
        st.markdown(f"**🧬 对应的数据输入格式：** `{datatype}`")

        if st.session_state.get("models_list"):
            with st.expander("📦 已有模型列表 (models_list)"):
                for model_name in st.session_state["models_list"]:
                    cols = st.columns([4, 1])
                    cols[0].markdown(f"- {model_name}")
                    if cols[1].button("🗑️ 删除", key=f"del_{model_name}"):
                        delete(st.session_state["final_model_type"], model_name)
                        st.session_state["model_list_changed"] = True

        if st.session_state.get("datasets_list"):
            with st.expander("🗂️ 已有数据集列表 (datasets_list)"):
                st.markdown("\n".join(f"- {item}" for item in st.session_state["datasets_list"]))
    final_model_type = st.session_state.final_model_type

    # ----------- 上传文件区域 -----------
    uploaded_zip = st.file_uploader(f"📦 上传模型文件包（model.zip），过大文件请手动解压缩并放入molplat_form/dataset/data/你选择的模型类型 目录下（molplat_form/dataset/data/{final_model_type}）", type=["zip"])
    if uploaded_zip:
        st.session_state["uploaded_model_zip"] = uploaded_zip
        st.success(f"✅ 上传模型包：{uploaded_zip.name}")

    uploaded_model_config = st.file_uploader("📄 上传模型配置文件（model_config.yaml）", type=["yaml"])
    if uploaded_model_config:
        st.session_state["uploaded_model_config"] = uploaded_model_config
        st.success(f"✅ 上传模型配置：{uploaded_model_config.name}")
        
    uploaded_data_config = st.file_uploader("📄 上传数据配置文件（data_config.yaml）", type=["yaml"])
    if uploaded_data_config:
        st.session_state["uploaded_data_config"] = uploaded_data_config
        st.success(f"✅ 上传数据配置：{uploaded_data_config.name}")

    uploaded_data_zip = st.file_uploader(f"🗂️ 上传数据文件包（data.zip），过大文件请手动解压缩并放入molplat_form/models/你选择的模型类型 目录下 （molplat_form/models/{final_model_type}）", type=["zip"])
    if uploaded_data_zip:
        st.session_state["uploaded_data_zip"] = uploaded_data_zip
        st.success(f"✅ 上传数据文件：{uploaded_data_zip.name}")

    # ----------- 显示用户输入状态 -----------
    if final_model_type:
        st.success(f"🎯 选择/输入的数据集类型：`{final_model_type}`")
    if st.button("🚀 提交并处理"):
        if st.session_state.model_type_select == "自定义输入" and not st.session_state.final_model_type.strip():
            st.warning("⚠️ 请输入自定义数据集类型名称后再提交。")
        else:# 获取上传的文件
            model_zip = st.session_state.get("uploaded_model_zip")
            model_config = st.session_state.get("uploaded_model_config")
            data_zip = st.session_state.get("uploaded_data_zip")
            data_config = st.session_state.get("uploaded_data_config")

            # 检查模型组是否完整
            model_ready = (model_zip is not None) and (model_config is not None)
            # 检查数据组是否完整
            data_ready = (data_zip is not None) and (data_config is not None)
            all_configs = (model_config is not None ) and (data_config is not None)

            # 情况1：模型组完整，data_zip 可以缺失（但 data_config 必须传）
            condition1 = model_ready and (data_config is not None)
            # 情况2：数据组完整，模型组可以完全缺失
            condition2 = data_ready and (not model_ready)
            condition3 = model_ready and data_ready

            if condition1 or condition2 or condition3 or all_configs:
                # ✅ 满足条件，调用 process
                result = process(
                    final_model_type,
                    model_zip,
                    model_config,
                    data_zip,
                    data_config
                )
                
                if result is True:
                    st.success("✅ 模型导入完成！")
                else:
                    st.error(result)
            else:
                # ❌ 不满足条件，提示错误
                missing = []
                if not model_ready:
                    missing.append("模型组（需同时上传 model_zip 和 model_config）")
                if data_config is None:
                    missing.append("data_config（必须上传）")
                if not data_ready and (data_zip is not None or data_config is not None):
                    missing.append("数据组不完整（需同时上传 data_zip 和 data_config）")

                st.error(f"""
                ⚠️ **提交失败！**  
                请确保符合以下条件之一：
                - **情况1**：完整上传模型组（`model_zip` + `model_config`），并至少上传 `data_config`（`data_zip` 可选），**或**  
                - **情况2**：完整上传数据组（`data_zip` + `data_config`），不上传模型组，**或** 
                - **情况3**: 全部完整上传 ，**或** 
                - **情况4**: 上传config 并将其余文件放入对应文件夹下


                """)
else:
    # ----------- 当 model_field 变化时，重置所有相关选择 -----------
    def on_model_field_change():
        st.session_state["selected_models"] = []
        st.session_state["selected_datasets"] = []
        st.session_state["selected_tasks"] = []
        st.session_state["_last_selected_dataset"] = None
        
    model_field_options = get_all_model_types()  # 按你的需求可扩展或自动加载

    # ✅ 添加模型特征字段选择控件
    st.selectbox(
        "模型所属数据集类型",
        options=model_field_options,
        key="selected_model_field",
        on_change=on_model_field_change
    )    
    # ----------- 从 model_dataset_map.yaml 获取数据集列表 -----------
    @st.cache_data
    def load_model_map(modelfield, path=MODEL_PATH):
        new_path = os.path.join(path, 'models.yaml')
        with open(new_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data.get(modelfield, {}).get("models", {})

    model_field = st.session_state["selected_model_field"]
    if model_field:
        model_options =[]
        model_map = load_model_map(model_field)
        for mode_l in model_map:
            submodels = get_submodel(model_field,mode_l)
            for submodel in submodels:
                full_model = f"{mode_l}_{submodel}"
                model_options.append(full_model)
        model_options_with_all = model_options + ["all"]
        # ----------- 记录模型选择前的值 -----------


        def on_model_change():
            st.session_state["selected_datasets"] = []
            st.session_state["selected_tasks"] = []

        # ✅ 多选控件（使用 session 保存 + 回调重置）
        st.multiselect(
            "模型名称(model)",
            options=model_options_with_all,
            key="selected_models",
            on_change=on_model_change
        )

    if "all" in st.session_state["selected_models"]:
        model = model_options
    else:
        model = st.session_state["selected_models"]
        

    if model:
        model_upper_list =[]
        for models in model:
            if isinstance(models, str) and "_" in models:
                model_part = models.split("_")[0]
            else:
                model_part = str(models).upper()
            if model_part not in model_upper_list:
                model_upper_list.append(model_part)
                readname = f"{model_part}_readme.md"
                outputname = f"{model_part}_output.py"
                dataname = f"{model_part}_data.py"
                modelname = f"{model_part}_model.py"
                reqname =  f"{model_part}_requirements.txt"
                READMEFILE_PATH = os.path.join(project_root, 'models',model_field,readname)
                OUTPUTFILE_PATH = os.path.join(project_root, 'models',model_field,model_part,outputname)
                DATAFILE_PATH = os.path.join(project_root, 'models',model_field,model_part,dataname)
                MODELFILE_PATH=os.path.join(project_root, 'models',model_field,model_part,modelname)
                REQ_PATH = os.path.join(project_root, 'models',model_field,reqname)
                show_file_selector(f"{model_part}: requirements.txt", REQ_PATH, is_text=True)
                show_update_button(model_part, reqname)
                show_create_button(model_part,reqname)
                show_file_selector(f"{model_part}: README.md", READMEFILE_PATH, is_markdown=True)
                show_file_selector(f"{model_part}: Output Script", OUTPUTFILE_PATH)
                show_file_selector(f"{model_part}: Data Script", DATAFILE_PATH)
                show_file_selector(f"{model_part}: Model Script", MODELFILE_PATH)
    #--------datasets 只有在 model 出现的时候再出现
    def on_dataset_change():
        st.session_state["selected_tasks"] = []  # 重置任务选择
        st.session_state["_last_selected_dataset"] = None  # 清除上次任务的缓存标记

    if "selected_datasets" not in st.session_state:
        st.session_state["selected_datasets"] = []

    if model:
        available_datasets = get_datasets_for_model(model, model_map)
        dataset_options_with_all = available_datasets + ["all"]

        st.multiselect(
            "数据集名称 (name)",
            options=dataset_options_with_all,
            key="selected_datasets",
            on_change=on_dataset_change
        )

        if "all" in st.session_state["selected_datasets"]:
            name = available_datasets
        else:
            name = st.session_state["selected_datasets"]

    # ----------- 任务选择（target_list）-----------
    if "selected_tasks" not in st.session_state:
        st.session_state["selected_tasks"] = []

    if "name" in locals() and name:
        if len(name) > 1:
            st.markdown("**任务名称 (target_list):** all")
            target_list = "all"
        else:
            dataset_name = name[0]

            try:
                utils = CheckUtils(st.session_state["selected_model_field"])
                available_tasks = utils.get_datasets_measure_names(dataset_name)
                task_options_with_all = available_tasks + ["all"]

                # 如果换了数据集，重置任务选择
                if st.session_state.get("_last_selected_dataset") != dataset_name:
                    st.session_state["selected_tasks"] = []
                    st.session_state["_last_selected_dataset"] = dataset_name

                st.multiselect(
                    "任务名称 (target_list)",
                    options=task_options_with_all,
                    key="selected_tasks"
                )

                if "all" in st.session_state["selected_tasks"]:
                    target_list = available_tasks
                else:
                    target_list = st.session_state["selected_tasks"]

            except Exception as e:
                st.warning(f"无法获取任务列表：{e}")
                target_list = "all"


    # ----------- evaluation 输入框 -----------
    if "eval" not in st.session_state:
        st.session_state["eval"] = True
    eval = st.checkbox("是否评估模型并绘图 (必须先上传数据)", key="eval")

    # ----------- smiles_list 输入框 -----------
    if "smiles_list" not in st.session_state:
        st.session_state["smiles_list"] = "random200"
    st.markdown("### 选择数据输入方式")
    mode_display_to_internal = {
        "自动评估(必须先上传对应数据)": "auto_eval",
        "上传文件": "file_upload",
        "手动输入": "manual_input"
    }
    mode_internal_to_display = {v: k for k, v in mode_display_to_internal.items()}

    # 控件：选择模式（只读，不直接改 session_state）
    selected_mode_display = st.radio(
        "请选择一种方式",
        options=list(mode_display_to_internal.keys()),
        index=list(mode_display_to_internal.values()).index(st.session_state["smiles_input_mode"])
    )

    # 将 radio 控件结果写入 session
    if mode_display_to_internal[selected_mode_display] != st.session_state["smiles_input_mode"]:
        st.session_state["smiles_input_mode"] = mode_display_to_internal[selected_mode_display]
        st.rerun()

    # 三种模式分别处理
    mode = st.session_state["smiles_input_mode"]

    if mode == "auto_eval":
        smiles_eval_mode = st.selectbox(
            "选择评估模式",
            ["random", "all"],
            index=["random", "all"].index(st.session_state["smiles_eval_mode"])
        )

        if smiles_eval_mode != st.session_state["smiles_eval_mode"]:
            st.session_state["smiles_eval_mode"] = smiles_eval_mode
            st.rerun()

        if st.session_state["smiles_eval_mode"] == "random":
            smiles_eval_num = st.number_input("请输入要随机选择的数量", min_value=1, value=st.session_state["smiles_eval_num"])
            if smiles_eval_num != st.session_state["smiles_eval_num"]:
                st.session_state["smiles_eval_num"] = smiles_eval_num
                st.session_state["smiles_list"] = f"random{smiles_eval_num}"
            else:
                st.session_state["smiles_list"] = f"random{st.session_state['smiles_eval_num']}"
        else:
            st.session_state["smiles_list"] = "all"

    elif mode == "file_upload":
        uploaded_file = st.file_uploader("上传包含 数据 的 .txt 或 .csv 文件", type=["txt", "csv"])
        if uploaded_file is not None:
            st.session_state["smiles_file"] = uploaded_file
            if uploaded_file.name.endswith(".txt"):
                content = uploaded_file.read().decode("utf-8")
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                st.session_state["smiles_list"] = lines
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                col = st.selectbox("选择数据所在列", df.columns)
                smiles = df[col].dropna().astype(str).tolist()
                st.session_state["smiles_list"] = smiles

    elif mode == "manual_input":
        text = st.text_area("请输入逗号分隔的数据", value=st.session_state["smiles_text_input"])
        if text != st.session_state["smiles_text_input"]:
            st.session_state["smiles_text_input"] = text
            smiles = [s.strip() for s in text.split(",") if s.strip()]
            st.session_state["smiles_list"] = smiles


    # ----------- 运行按钮 -----------
    if st.button("运行模型配置并保存配置文件"):
        fields_to_convert = ["model", "name", "target_list"]
        config = load_config()
        config["user_argument"] = st.session_state["selected_model_field"]
        config["model"] = model
        config["name"] = name
        config["target_list"] = target_list
        config["eval"] = st.session_state["eval"]
        smiles_val = st.session_state.get("smiles_list", "")
        if isinstance(smiles_val, list):
            config["smiles_list"] = ",".join(smiles_val)
        else:
            config["smiles_list"] = smiles_val
        config = list_to_csv_fields(config, fields_to_convert)

        save_config(config)
        st.success("配置已保存！")

        try:
            # 运行子进程并捕获输出
            result = subprocess.run(
                ["python", RUN_SCRIPT_PATH],
                capture_output=True,  # 捕获标准输出和错误输出
                text=True,            # 以文本形式返回
                encoding='utf-8',     # 指定编码
                check=True            # 如果返回非零状态码则引发异常
            )
            st.success("✅ 模型运行完成！")
            
            # 处理成功运行后的逻辑...
            result_path = os.path.join(project_root,'results','results')
            run_id,latest_run_path = get_latest_run_folder(result_path)
            history_record = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "model_argument": config["user_argument"],
                "model": config["model"],
                "dataset": config["name"],
                "task": config["target_list"],
                "data": config["smiles_list"],
                "eval": config["eval"]
            }
            history_list = []
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            history_list.insert(0, history_record)
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(history_list, f, indent=2, ensure_ascii=False)

            if latest_run_path:
                if config['eval']:
                    plot_dir = os.path.join(latest_run_path, "plots")
                    st.markdown("## 🖼️ 模型分析图 (plots)")
                    display_images_recursively(plot_dir)

                st.markdown("## 📊 模型结果表格 (CSVs)")
                display_csv_tables(latest_run_path)
            else:
                st.warning("未找到任何 runXX 结果目录。")

        except subprocess.CalledProcessError as e:
            # 当命令返回非零状态码时，显示详细错误
            error_msg = f"❌ 模型运行失败 (返回码: {e.returncode})!\n\n" 
            error_msg += "=== 错误详情 ===\n"
            error_msg += e.stderr + "\n"
            error_msg += "请检查模型环境是否正确配置 （model: README.md）"
            st.error(error_msg)
            
            # 在终端打印完整错误（用于调试）
            print("="*80)
            print(f"子进程详细错误信息 (返回码 {e.returncode}):")
            print(e.stderr)
            print("="*80)
            
        except Exception as e:
            # 其他异常
            st.error(f"运行出错：{e}")
            print(f"运行出错：{e}")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history_list = json.load(f)

        if history_list:
            st.markdown("---")
            st.markdown("### 📂 历史运行记录（可以在results/results下查看每一次的具体结果）")
            history_labels = [f"{h['run_id']} | 数据集类型：{h['model_argument']}|模型: {h['model']} | 数据集: {h['dataset']} | 任务: {h['task']}| 数据:{h['data']}" for h in history_list]
            selected_index = st.selectbox("选择历史记录运行 ID 以查看结果：", options=list(range(len(history_list))), format_func=lambda i: history_labels[i])

            selected = history_list[selected_index]
            selected_run_path = os.path.join(project_root, 'results', 'results', selected["run_id"])

            if os.path.exists(selected_run_path):
                if selected.get("eval", True):
                    st.markdown("## 🖼️ 模型分析图 (plots)")
                    display_images_recursively(os.path.join(selected_run_path, "plots"))

                st.markdown("## 📊 模型结果表格 (CSVs)")
                display_csv_tables(selected_run_path)
            else:
                st.warning("找不到对应的历史目录。")
        
            
    

import streamlit as st
import yaml
import os
import sys
import subprocess
import pathlib
import pandas as pd
import re
import json
import shutil
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
st.markdown("**一站式AI化学平台** - 模型与数据管理、兼容环境搭建、智能预测评估、可视化分析")

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
def get_envs():
    env_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../environment.yaml')

    try:
        with open(env_root, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return {}

    if not isinstance(data, dict):
        print("文件内容格式异常，期望顶层为字典")
        return {}

    # 最高级键和对应所有次级键
    result = {}
    for top_key, sub_dict in data.items():
        if isinstance(sub_dict, dict):
            result[top_key] = list(sub_dict.keys())
        else:
            result[top_key] = []

    return result
     




# ----------- 初始化 session_state -----------
if "selected_model_field" not in st.session_state:      # dataset_type
    st.session_state["selected_model_field"] = None
if "selected_model_workflow" not in st.session_state:   # workflow_type
    st.session_state["selected_model_workflow"] = None
if "selected_model_names" not in st.session_state:     # model_type
    st.session_state["selected_model_names"] = []
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
def on_workflow_change():
    st.session_state["selected_model_names"] = []
def get_top_level_keys():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        return list(data.keys())
    else:
        return []
def update(file, envname, model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(current_dir, '../env_utils.py'))
    env_md_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    cmd = [sys.executable, script_path, "update", '-r', file, '-e', envname]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.success("更新执行成功")
    except subprocess.CalledProcessError as e:
        st.error(f"更新执行失败，返回码：{e.returncode}")
        return False

    try:
        with open(env_md_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        if envname not in data:
            st.error(f"错误: environment.yaml 顶层找不到环境名 '{envname}'")
            return False

        data[envname][model] = file

        with open(env_md_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)

        return True
    except Exception as e:
        st.error(f"写入 environment.yaml 失败: {e}")
        return False


def create(model, file, envname, version):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(current_dir, '../env_utils.py'))
    base_reqs = os.path.abspath(os.path.join(current_dir, '../requirements.txt'))
    env_md_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))

    cmd = [sys.executable, script_path, 'create', '-r', base_reqs, '-a', file, '-e', envname, '-p', version]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.success("创建执行成功")
    except subprocess.CalledProcessError as e:
        st.error(f"创建执行失败，返回码：{e.returncode}")
        return False

    try:
        with open(env_md_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        if envname not in data or not isinstance(data[envname], dict):
            data[envname] = {}

        data[envname][model] = file
        data[envname]['molplat'] = "requirements.txt"

        with open(env_md_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)

        return True
    except Exception as e:
        st.error(f"写入 environment.yaml 失败: {e}")
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
                st.text("如需重新查看环境列表，请手动刷新 'ctrl r'")
            else:
                st.error("❌ Update 失败，请检查输出信息")


def show_create_button(model, reqname):
    with st.expander("创建环境"):
        st.markdown("### 创建模型配置")

        col3, col4 = st.columns(2)

        with col3:
            py_version = st.text_input("Python 版本", value="3.11.8", max_chars=10)

        with col4:
            env_name = st.text_input("环境名字", max_chars=20)

        if st.button("Create"):
            if not py_version.strip() or not env_name.strip() or not model.strip() or not reqname.strip():
                st.error("请填写所有字段，包括模型名、依赖文件、Python 版本和环境名！")
            else:
                st.text("创建环境中⏳")
                success = create(model, reqname, env_name, py_version)
                if success:
                    st.success(f"Create 调用成功，环境名={env_name}, Python版本={py_version}")
                    st.text("创建新环境成功，使用新环境，请关闭重启平台，输入新环境名")
                else:
                    st.error("创建环境失败，请查看上方错误信息。")

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
def repair_environment_record():
    try:
        # 获取当前系统中所有conda环境
        conda_envs = get_conda_environments()
        
        # 读取environment.yaml文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_md_path = os.path.abspath(os.path.join(current_dir, '../environment.yaml'))
        
        with open(env_md_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # 检查并移除不存在于系统的环境
        original_count = len(data)
        keys_to_remove = [env for env in data if env not in conda_envs]
        keys_to_keep = [env for env in data if env not in keys_to_remove]
        for env in keys_to_remove:
            del data[env]
        
        # 保存更新后的文件
        with open(env_md_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
        
        return True, len(keys_to_remove), [keys_to_remove,keys_to_keep]
    except Exception as e:
        st.error(f"修复失败: {e}")
        return False, 0, []

# 获取系统中所有conda环境
def get_conda_environments():
    try:
        # 使用conda命令获取环境列表
        result = subprocess.run(
            ['conda', 'env', 'list', '--json'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析JSON输出
        env_data = json.loads(result.stdout)
        envs = env_data.get('envs', [])
        
        # 提取环境名称（路径的最后部分）
        env_names = set()
        for env_path in envs:
            # 基本环境通常是第一个，名称为"base"
            if env_path == env_data.get('root_prefix'):
                env_names.add('base')
            else:
                env_name = os.path.basename(env_path)
                env_names.add(env_name)
        
        return env_names
    except Exception as e:
        st.error(f"获取conda环境失败: {e}")
        return set()
    
close_tab_js = """
<script>
    window.close();
</script>
"""
exit_col_space, exit_col_btn = st.columns([9, 1])
with exit_col_btn:
    if st.button("❌退出"):
        st.warning("程序即将关闭...")
        st.components.v1.html(close_tab_js)
        os._exit(0)


col1, col2 = st.columns([10, 2])
with col1:
    envs = get_envs()

    # 通过 HTML 和 CSS 控制标题字体大小
    st.markdown("""
        <style>
        .small-title {
            font-size: 20px;
            font-weight: bold;
        }
        .env-item {
            margin-bottom: 8px;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 平台已创建环境列表，对应该环境支持的依赖模块名 （如molplat为平台基础依赖）")
    
    # 显示环境和次级键
    for top_key, sub_keys in envs.items():
        sub_keys_str = ", ".join(sub_keys) if sub_keys else "(无依赖安装)"
        st.markdown(f'<div class="env-item"><b>{top_key}</b>: {sub_keys_str}</div>', unsafe_allow_html=True)


    current_env = os.environ.get('CONDA_DEFAULT_ENV', '未检测到当前环境')

    st.markdown(f"<div style='font-size:14px;'>当前平台工作环境：{current_env}</div>", unsafe_allow_html=True)
    # 添加环境修复按钮
    st.markdown("---")
    with st.expander("🔧 修复环境记录", expanded=False):
        st.markdown("**扫描并移除系统中已不存在的环境记录**")
        st.warning("此操作将更新 environment.yaml 文件，移除所有不存在的环境记录")
        
        if st.button("扫描并修复环境记录"):
            st.text("⏳ 正在扫描环境...")
            success, removed_count, return_list = repair_environment_record()
            if success:
                if removed_count > 0:
                    st.error(f"发现 {removed_count} 个不存在环境记录:")
                    st.error(",".join(return_list[0]))
                    st.success("✅已移除无效环境")
                    st.success(f"有效环境:")
                    st.success(",".join(return_list[1]))
                    st.text("如需更新环境列表，请手动刷新 'ctrl r'")
                else:
                    st.info("未检测到已删除环境记录，环境列表正常")
            else:
                st.error("❌ 修复失败，请检查输出信息")

    st.write("")
    st.write("")

with col2:
    if st.button("➕ 添加数据集与模型（点击以返回）"):
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
        st.session_state["selected_model_workflows"] = []  # 改为列表
        st.session_state["selected_model_names"] = []
        st.session_state["selected_datasets"] = []
        st.session_state["selected_tasks"] = []
        st.session_state["_last_selected_dataset"] = None
        
    model_field_options = get_all_model_types()
    st.markdown("### 请选择平台预测方法")
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

        # 返回数据结构改为 {工作流: {模型名称: 配置}}
        return data.get(modelfield, {}).get("models", {})

    model_field = st.session_state["selected_model_field"]
    if model_field:
        model_map = load_model_map(model_field)
        workflows = list(model_map.keys())  # 获取所有工作流
        
        # 将工作流选择改为多选框
        st.multiselect(
            "模型工作流（可多选）",
            options=workflows,
            key="selected_model_workflows",  # 改为复数形式
            on_change=lambda: st.session_state.update({"selected_model_names": []})  # 工作流变化时重置模型选择
        )
        
        # 根据选择的多个工作流加载所有模型
        if st.session_state["selected_model_workflows"]:
            all_model_options = []
            
            # 遍历每个选中的工作流
            for workflow in st.session_state["selected_model_workflows"]:
                # 获取该工作流下的所有模型
                for model_key in model_map[workflow].keys():
                    full_model = f"{workflow}_{model_key}"
                    all_model_options.append(full_model)
            
            # 去重并排序
            all_model_options = sorted(set(all_model_options))
            model_options_with_all = all_model_options + ["all"]
            
            st.multiselect(
                "模型名称（可多选）",
                options=model_options_with_all,
                key="selected_model_names"
            )

    # 使用 selected_model_names 替代原来的 selected_models
    if "all" in st.session_state["selected_model_names"]:
        model = all_model_options  # 使用之前收集的所有模型
    else:
        model = st.session_state["selected_model_names"]
        
    # 环境管理部分：为每个选中的工作流分别显示
    if model and st.session_state["selected_model_workflows"]:
        # 遍历每个选中的工作流
        for model_workflow in st.session_state["selected_model_workflows"]:
            
            readname = f"{model_workflow}_readme.md"
            outputname = f"{model_workflow}_output.py"
            dataname = f"{model_workflow}_data.py"
            modelname = f"{model_workflow}_model.py"
            reqname =  f"{model_workflow}_requirements.txt"
            
            # 构建文件路径
            READMEFILE_PATH = os.path.join(project_root, 'models', model_field, readname)
            OUTPUTFILE_PATH = os.path.join(project_root, 'models', model_field, model_workflow, outputname)
            DATAFILE_PATH = os.path.join(project_root, 'models', model_field, model_workflow, dataname)
            MODELFILE_PATH = os.path.join(project_root, 'models', model_field, model_workflow, modelname)
            REQ_PATH = os.path.join(project_root, 'models', model_field, reqname)
            
            st.markdown("#### 环境管理功能")
            st.markdown("**本功能默认使用模型工作流requirements.txt文件，使用一建化功能前，请查阅README.md，检查是否为模型工作流所需全部依赖，部分依赖可能须按指引手动安装**")
            
            # 显示文件选择器
            show_file_selector(f"{model_workflow}: requirements.txt ", REQ_PATH, is_text=True)
            show_file_selector(f"{model_workflow}: README.md", READMEFILE_PATH, is_markdown=True)
            
            # 显示环境管理按钮
            col1, col2 = st.columns(2)
            with col1:
                show_update_button(model_workflow, REQ_PATH)
            with col2:
                show_create_button(model_workflow, REQ_PATH)
            
            st.markdown("**模型工作流核心文件**")
            show_file_selector(f"{model_workflow}: Output Script", OUTPUTFILE_PATH)
            show_file_selector(f"{model_workflow}: Data Script", DATAFILE_PATH)
            show_file_selector(f"{model_workflow}: Model Script", MODELFILE_PATH)
            
            st.markdown("---")  # 添加分隔线

    #--------datasets 只有在 model 出现的时候再出现
    def on_dataset_change():
        st.session_state["selected_tasks"] = []  # 重置任务选择
        st.session_state["_last_selected_dataset"] = None  # 清除上次任务的缓存标记

    if "selected_datasets" not in st.session_state:
        st.session_state["selected_datasets"] = []

    if model:
        available_datasets = get_datasets_for_model(model, model_map)
        dataset_options_with_all = available_datasets + ["all"]
        st.markdown("### 请选择预测对象")
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
            smiles_eval_num = st.number_input("请输入要随机选择的数量", min_value=1, value=st.session_state["smiles_eval_num"], step=200)
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
                config_path = os.path.join(latest_run_path, "config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
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
        
        # 添加修复历史记录选项
        st.markdown("#### 🔧 修复历史记录")
        col_repair1, col_repair2 , col_repair3= st.columns(3)
        
                # 修改移除无效记录功能
        with col_repair1:
            if st.button("移除无效记录", key="remove_invalid"):
                # 扫描结果目录获取有效run_id
                valid_run_ids = set()
                results_dir = os.path.join(project_root, 'results', 'results')
                if os.path.exists(results_dir):
                    for run_id in os.listdir(results_dir):
                        run_path = os.path.join(results_dir, run_id)
                        if os.path.isdir(run_path):
                            # 检查结果目录是否包含配置文件
                            if os.path.exists(os.path.join(run_path, 'config.json')):
                                valid_run_ids.add(run_id)
                
                # 过滤历史记录，只保留有效记录
                updated_history = [r for r in history_list if r['run_id'] in valid_run_ids]
                
                # 保存更新后的历史记录
                with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(updated_history, f, indent=2, ensure_ascii=False)
                
                st.success(f"已移除 {len(history_list) - len(updated_history)} 条无效记录！")
                st.rerun()
                # 新增第三列：清除全部历史记录

        with col_repair2:
            if st.button("添加缺失记录", key="add_missing"):
                # 扫描结果目录获取所有run_id
                results_dir = os.path.join(project_root, 'results', 'results')
                existing_run_ids = set(r['run_id'] for r in history_list)
                new_records = []
                
                if os.path.exists(results_dir):
                    for run_id in os.listdir(results_dir):
                        if run_id in existing_run_ids:
                            continue
                            
                        run_path = os.path.join(results_dir, run_id)
                        if not os.path.isdir(run_path):
                            continue
                            
                        # 检查配置文件是否存在
                        config_path = os.path.join(run_path, 'config.json')
                        if os.path.exists(config_path):
                            try:
                                with open(config_path, 'r', encoding='utf-8') as config_file:
                                    run_config = json.load(config_file)
                                    
                                # 获取目录创建时间作为时间戳
                                ctime = os.path.getctime(run_path)
                                timestamp = datetime.fromtimestamp(ctime).isoformat()
                                
                                # 创建记录 - 使用config.json中的参数
                                new_records.append({
                                    'timestamp': timestamp,
                                    'run_id': run_id,
                                    'model_argument': run_config.get("user_argument", "未知"),
                                    'model': run_config.get("model", "未知"),
                                    'dataset': run_config.get("name", "未知"),
                                    'task': run_config.get("target_list", "未知"),
                                    'data': run_config.get("smiles_list", "未知"),
                                    'eval': run_config.get("eval", True)
                                })
                            except Exception as e:
                                st.warning(f"无法读取 {run_id} 的配置文件: {e}")
                        else:
                            # 如果没有配置文件，创建基础记录
                            ctime = os.path.getctime(run_path)
                            timestamp = datetime.fromtimestamp(ctime).isoformat()
                            new_records.append({
                                'timestamp': timestamp,
                                'run_id': run_id,
                                'model_argument': '未知',
                                'model': '未知',
                                'dataset': '未知',
                                'task': '未知',
                                'data': '未知',
                                'eval': True
                            })
                
                if new_records:
                    # 添加新记录到历史记录
                    updated_history = history_list + new_records
                    
                    # 保存更新后的历史记录
                    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                        json.dump(updated_history, f, indent=2, ensure_ascii=False)
                    
                    st.success(f"已添加 {len(new_records)} 条缺失记录！")
                    st.rerun()
                else:
                    st.info("未发现缺失记录")

        with col_repair3:
            if st.button("清除全部历史记录", key="clear_all_history", 
                         help="⚠️ 清除所有历史记录（不会删除结果文件）"):
                if st.session_state.get("confirm_clear_all", False):
                    # 删除历史记录文件
                    try:
                        os.remove(HISTORY_PATH)
                        st.success("已清除全部历史记录！")
                        st.session_state.pop("confirm_clear_all", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"清除失败: {e}")
                else:
                    st.session_state["confirm_clear_all"] = True
                    st.warning("确定要清除全部历史记录吗？再次点击按钮确认。")
        
        # 为每条记录创建一行
        for i, record in enumerate(history_list):
            # 创建一行布局
            col_info, col_view, col_delete = st.columns([8, 1, 1])
            
            # 左侧：显示记录信息
            with col_info:
                st.markdown(f"**{record['run_id']}** | 数据集类型：{record['model_argument']}|模型: {record['model']} | 数据集: {record['dataset']} | 任务: {record['task']}| 数据:{record['data']}")
            
            # 中间：查看结果按钮
            with col_view:
                view_key = f"view_{record['run_id']}"
                if st.button("查看结果", key=view_key):
                    # 切换查看状态
                    st.session_state[f"show_{record['run_id']}"] = not st.session_state.get(f"show_{record['run_id']}", False)
            
            # 右侧：删除按钮
            with col_delete:
                delete_key = f"delete_{record['run_id']}"
                if st.button("🗑️", key=delete_key, help="删除此记录"):
                    # 确认删除
                    if st.session_state.get(f"confirm_delete_{record['run_id']}", False):
                        # 删除结果文件夹
                        run_folder = os.path.join(project_root, 'results', 'results', record['run_id'])
                        if os.path.exists(run_folder):
                            try:
                                shutil.rmtree(run_folder)
                                st.success(f"已删除结果文件夹: {run_folder}")
                            except Exception as e:
                                st.error(f"删除文件夹失败: {e}")
                        
                        # 从历史记录中移除
                        del history_list[i]
                        
                        # 保存更新后的历史记录
                        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                            json.dump(history_list, f, indent=2, ensure_ascii=False)
                        
                        st.success("历史记录已删除！")
                        st.rerun()
                    else:
                        # 设置确认标志
                        st.session_state[f"confirm_delete_{record['run_id']}"] = True
                        st.warning("确定要删除这条记录吗？再次点击删除按钮确认。")
            
            # 显示结果区域（如果该记录被展开）
            if st.session_state.get(f"show_{record['run_id']}", False):
                selected_run_path = os.path.join(project_root, 'results', 'results', record["run_id"])
                
                if os.path.exists(selected_run_path):
                    if record.get("eval", True):
                        st.markdown("#### 🖼️ 模型分析图")
                        display_images_recursively(os.path.join(selected_run_path, "plots"))

                    st.markdown("#### 📊 模型结果表格")
                    display_csv_tables(selected_run_path)
                else:
                    st.warning("找不到对应的历史目录。")
                
                st.markdown("---")
        
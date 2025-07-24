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
from process import process
try:
    project_root = pathlib.Path(__file__).resolve().parents[1]
except NameError:
    project_root = pathlib.Path(os.getcwd()).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from models.check_utils import get_datasets_measure_names,CheckUtils
from streamlit_option_menu import option_menu

def set_streamlit_upload_limit(limit_mb=2048):
    config_dir = os.path.expanduser("~/.streamlit")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")

    with open(config_path, "w") as f:
        f.write(f"[server]\nmaxUploadSize = {limit_mb}\n")

set_streamlit_upload_limit(2048)

st.set_page_config(layout="wide")
st.title("分子性质预测集成平台")
st.markdown("根据模型类型自动加载数据集，仅在需要时显示额外参数，最终保存为配置文件并可供模型运行。")

# ----------- 配置路径 -----------
MODEL_PATH =os.path.join(project_root,'models')
CONFIG_PATH = os.path.join(project_root,'result_analysis','config_run.yaml')
# MODEL_MAP_PATH = os.path.join(project_root,'models','model_datasets.yaml')
RUN_SCRIPT_PATH = os.path.join(project_root,'result_analysis','run_all.py')
HISTORY_PATH = os.path.join(project_root, 'results', 'results','run_history,json')
MODEL_DATASET_PATH = os.path.join(MODEL_PATH,'models.yaml')




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
    for model_type, sub_models in models_config.items():
        for sub_model in sub_models:
            model_names.append(f"{model_type}_{sub_model}")
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

        

def show_file_selector(label, file_path, is_markdown=False, height=500):
    """显示复选框，勾选后展示带固定高度滚动条的文件内容"""
    if not os.path.exists(file_path):
        st.write(f"{label} 文件不存在：{file_path}")
        return

    show_content = st.checkbox(f"显示 {label}", key=f"show_{label}")

    if show_content:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if is_markdown:
            st.markdown(content)
        else:
            # st.code 支持设置 height，显示带滚动条的代码区域
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

def process(dataset_type,zip):
    return



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

# ----------- 展开按钮 -----------
col1, col2 = st.columns([10, 1])
with col2:
    if st.button("➕ 添加模型类型"):
        st.session_state["show_model_input"] = not st.session_state["show_model_input"]

# ----------- 展开区域 -----------
if st.session_state["show_model_input"]:
    st.markdown("#### 🔧 自定义模型类型与模型包上传")

    try:
        all_model_types = get_all_model_types()
    except Exception as e:
        st.warning(f"加载模型类型失败：{e}")
        all_model_types = []

    # 修改后的选择控件
    model_type_options = ["自定义输入"] + all_model_types
    current_index = model_type_options.index(
        st.session_state["final_model_type"] 
        if st.session_state["final_model_type"] in model_type_options 
        else "自定义输入"
    )

    # 主选择框 - 直接绑定到 session_state
    selected_option = st.selectbox(
        "从已有模型类型中选择或直接输入新类型：",
        options=model_type_options,
        index=current_index,
        key="model_type_select"  # 直接使用key绑定
    )

    # 根据选择显示自定义输入框或模型信息
    if st.session_state.model_type_select == "自定义输入":
        st.text_input(
            "请输入新的模型类型",
            value=st.session_state.final_model_type,
            key="custom_model_input"  # 直接使用key绑定
        )
        # 立即更新final_model_type
        st.session_state.final_model_type = st.session_state.custom_model_input
    else:
        st.session_state.final_model_type = st.session_state.model_type_select
        # 显示模型信息（保持不变）
        datatype = get_data_type(st.session_state.final_model_type)
        st.markdown(f"**🧬 模型输入格式：** `{datatype}`")
        models_list, datasets_list = get_models_and_data(st.session_state.final_model_type)
        
        if models_list:
            with st.expander("📦 已有模型列表 (models_list)"):
                st.markdown("\n".join(f"- {item}" for item in models_list))
        if datasets_list:
            with st.expander("🗂️ 已有数据集列表 (datasets_list)"):
                st.markdown("\n".join(f"- {item}" for item in datasets_list))
    final_model_type = st.session_state.final_model_type

    # ----------- 上传文件区域 -----------
    uploaded_zip = st.file_uploader("📦 上传模型文件包（model.zip）", type=["zip"])
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

    uploaded_data_zip = st.file_uploader("🗂️ 上传数据文件包（data.zip）", type=["zip"])
    if uploaded_data_zip:
        st.session_state["uploaded_data_zip"] = uploaded_data_zip
        st.success(f"✅ 上传数据文件：{uploaded_data_zip.name}")

    # ----------- 显示用户输入状态 -----------
    if final_model_type:
        st.success(f"🎯 选择/输入的模型类型：`{final_model_type}`")
    if st.button("🚀 提交并处理模型类型"):
        # 获取上传的文件
        model_zip = st.session_state.get("uploaded_model_zip")
        model_config = st.session_state.get("uploaded_model_config")
        data_zip = st.session_state.get("uploaded_data_zip")
        data_config = st.session_state.get("uploaded_data_config")

        # 检查模型组是否完整
        model_ready = (model_zip is not None) and (model_config is not None)
        # 检查数据组是否完整
        data_ready = (data_zip is not None) and (data_config is not None)

        # 情况1：模型组完整，data_zip 可以缺失（但 data_config 必须传）
        condition1 = model_ready and (data_config is not None)
        # 情况2：数据组完整，模型组可以完全缺失
        condition2 = data_ready and (not model_ready)
        condition3 = model_ready and data_ready

        if condition1 or condition2:
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
            - **情况3**: 全部完整上传 


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
        "模型输入特征类型 (model_field)",
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
            "模型类型 (model)",
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
                READMEFILE_PATH = os.path.join(project_root, 'models',model_field,readname)
                OUTPUTFILE_PATH = os.path.join(project_root, 'models',model_field,model_part,outputname)
                DATAFILE_PATH = os.path.join(project_root, 'models',model_field,model_part,dataname)
                MODELFILE_PATH=os.path.join(project_root, 'models',model_field,model_part,modelname)
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
                col = st.selectbox("选择 SMILES 所在列", df.columns)
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
            result = subprocess.run(
                ["conda", "run", "-n", "molplat", "python", RUN_SCRIPT_PATH],
                check=True  # 自动抛出异常如果失败
            )
            st.success("✅ 模型运行完成！")
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

        except subprocess.CalledProcessError:
            st.error("❌ 模型运行失败！")
        except Exception as e:
            st.error(f"运行出错：{e}")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history_list = json.load(f)

        if history_list:
            st.markdown("---")
            st.markdown("### 📂 历史运行记录")
            history_labels = [f"{h['run_id']} | 模型: {h['model']} | 数据集: {h['dataset']} | 任务: {h['task']}| 数据:{h['data']}" for h in history_list]
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
        
            
    

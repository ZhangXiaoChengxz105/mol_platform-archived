import yaml
import sys
import os
import zipfile
import shutil
from pathlib import Path
import streamlit as st
import pprint
import stat

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH,'dataset','data')

class InlineListDumper(yaml.SafeDumper):
    pass

# 自定义 list representer：行内格式，list 中所有元素都用单引号
def _represent_inline_list(dumper, data):
    seq = []
    for item in data:
        if isinstance(item, str):
            node = dumper.represent_scalar('tag:yaml.org,2002:str', item, style="'")
        else:
            node = dumper.represent_data(item)
        seq.append(node)
    return yaml.nodes.SequenceNode('tag:yaml.org,2002:seq', seq, flow_style=True)

# 注册 list 的 representer
InlineListDumper.add_representer(list, _represent_inline_list)

# 保存函数
def save_fixed_config_yaml(config: dict, path: str):
    """保存 config 到指定路径，list 使用行内格式，元素用单引号包裹"""
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, Dumper=InlineListDumper)

def load_yaml_from_streamlit(uploaded_file):
    """
    从Streamlit上传文件加载配置，可以兼容 Python dict 风格或 YAML 格式
    """
    if uploaded_file is None:
        return {}

    try:
        content = uploaded_file.getvalue().decode("utf-8")

        # 尝试 YAML 解析
        try:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass

        # 尝试 Python dict 风格（eval）
        try:
            from ast import literal_eval
            parsed = literal_eval(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            st.error(f"⚠️ 无法解析配置文件（YAML 和 Python 字典格式都失败）：{e}")
            return {}

    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")
        return {}

def merge_yaml_configs(config1, config2):
    merged = {}

    # 合并 config: 合并 key，并对值（list）去重
    merged_config = dict(config1.get('config', {}))
    for k, v in config2.get('config', {}).items():
        if k in merged_config:
            merged_config[k] = list(set(merged_config[k]) | set(v))
        else:
            merged_config[k] = v
    merged['config'] = merged_config

    # data_type: 优先使用 config2
    merged['data_type'] = config2.get('data_type', config1.get('data_type', 'smiles'))

    # dataset_names 合并去重
    merged['dataset_names'] = list(
        set(config1.get('dataset_names', [])) | set(config2.get('dataset_names', []))
    )

    # regression_datasets 合并去重
    merged['regression_datasets'] = list(
        set(config1.get('regression_datasets', [])) | set(config2.get('regression_datasets', []))
    )

    return merged
def merge_model_configs(config1, config2):
    """
    合并 model 配置：
    - 顶层 key 一样时合并 datasets（交集）和 models（一级键冲突则 config2 覆盖）
    - 顶层 key 不同则直接添加 config2 的新键
    """
    if not isinstance(config1, dict): config1 = {}
    if not isinstance(config2, dict): config2 = {}

    merged_config = config1.copy()

    for top_key in config2:
        if top_key in merged_config:
            # 合并 datasets（交集）
            datasets1 = set(merged_config[top_key].get("datasets", []))
            datasets2 = set(config2[top_key].get("datasets", []))
            merged_datasets = list(datasets1 & datasets2)

            # 合并 models：只处理一级键，重复的整体替换
            models1 = merged_config[top_key].get("models", {})
            models2 = config2[top_key].get("models", {})
            merged_models = models1.copy()

            for model_type in models2:
                merged_models[model_type] = models2[model_type]  # ⬅️ 覆盖或新增一级键

            merged_config[top_key] = {
                "datasets": merged_datasets,
                "models": merged_models
            }
        else:
            # 新顶层键，直接加进来
            merged_config[top_key] = config2[top_key]

    return merged_config

def save_config_as_python_style(config: dict, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        # 写基本字段
        f.write(f"data_type: {repr(config.get('data_type', 'smiles'))}\n")
        f.write(f"dataset_names: {repr(config.get('dataset_names', []))}\n")
        f.write(f"regression_datasets: {repr(config.get('regression_datasets', []))}\n\n")

        # 写 config 字段，换行美化
        f.write("config: {\n")
        for k, v in config.get('config', {}).items():
            f.write(f"    {repr(k)}: [\n")
            for item in v:
                f.write(f"        {repr(item)},\n")
            f.write("    ],\n")
        f.write("}\n")
def open_and_merge_data_yaml(model_type,data_config):
    DATA_CONFIG_DIR = os.path.join(DATA_PATH,model_type,'dataset.yaml')
    new_config = load_yaml_from_streamlit(data_config)
    if os.path.exists(DATA_CONFIG_DIR):
        with open(DATA_CONFIG_DIR,'r') as f:
            old_config = yaml.safe_load(f) or {}
        merged = merge_yaml_configs(old_config,new_config)
        save_config_as_python_style(merged,DATA_CONFIG_DIR)
    else:
        save_config_as_python_style(new_config,DATA_CONFIG_DIR)
        
def process_data(model_type, data_config, data_zip,processed_data):
    D_R = os.path.join(DATA_PATH, model_type)
    os.makedirs(D_R, exist_ok=True)
    temp_extract_dir = os.path.join(D_R, "temp_extract")

    try:
        # 清空临时目录（如果存在）
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

        # 解压 ZIP 文件到临时目录
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # 获取临时目录下的所有内容
        top_level_items = os.listdir(temp_extract_dir)
        top_level_dirs = [d for d in top_level_items if os.path.isdir(os.path.join(temp_extract_dir, d))]

        # 判断 source_dir 选择逻辑
        if len(top_level_dirs) == 1:
            source_dir = os.path.join(temp_extract_dir, top_level_dirs[0])
        else:
            # 默认按 model_type 命名的目录
            source_dir = os.path.join(temp_extract_dir, model_type)

        # 检查 source_dir 是否存在
        if not os.path.exists(source_dir):
            return f"解压失败：未找到目录 {source_dir}"

        # 移动所有文件到 D_R
        for root, _, files in os.walk(source_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, source_dir)
                dst_path = os.path.join(D_R, rel_path)

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)

        # 合并配置文件
        if not processed_data:
            open_and_merge_data_yaml(model_type, data_config)
        return True

    except Exception as e:
        return f"解压失败: {e}"

    finally:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
            
    
def process_model(model_type,model_config,model_zip,data_config,processed_data,processed_model):
    M_D = os.path.join(ROOT_PATH,'models')
    NEW_D =os.path.join(M_D,model_type)
    new_config = load_yaml_from_streamlit(model_config)
    MODEL_CONFIG_DIR =os.path.join(M_D,'models.yaml')
    OTHER_CONFIG_DIR = os.path.join(M_D,model_type,'models.yaml')
    os.makedirs(NEW_D, exist_ok=True)
    if not processed_model:
        if os.path.exists(MODEL_CONFIG_DIR):
            with open(MODEL_CONFIG_DIR,'r') as f:
                old_config = yaml.safe_load(f) or {}
            merged = merge_model_configs(old_config,new_config)
            save_fixed_config_yaml(merged,MODEL_CONFIG_DIR)
            section = {k: v for k, v in merged.items() if k == model_type}
            if os.path.exists(OTHER_CONFIG_DIR):
                with open(OTHER_CONFIG_DIR,'r') as f:
                    old_section_config = yaml.safe_load(f) or {}
                    section_merged = merge_model_configs(old_section_config,section)
                    save_fixed_config_yaml(section_merged,OTHER_CONFIG_DIR)
            else:
                save_fixed_config_yaml(section,OTHER_CONFIG_DIR)

        else:
            save_fixed_config_yaml(new_config,MODEL_CONFIG_DIR)
            save_fixed_config_yaml(new_config,OTHER_CONFIG_DIR)

    if not processed_data:
        # DATA_CONFIG_DIR = os.path.join(DATA_PATH,model_type,'dataset.yaml')
        # new_config = load_yaml_from_streamlit(data_config)
        # if os.path.exists(DATA_CONFIG_DIR):
        #     with open(DATA_CONFIG_DIR,'r') as f:
        #         old_config = yaml.safe_load(f) or {}
        #     merged = merge_yaml_configs(old_config,new_config)
        #     with open(DATA_CONFIG_DIR, 'w') as f:
        #         yaml.dump(merged, f)
        # else:
        #     with open(DATA_CONFIG_DIR, 'w') as f:
        #         yaml.dump(new_config, f)
        open_and_merge_data_yaml(model_type,data_config)
    D_R = os.path.join(ROOT_PATH,'models', model_type)
    os.makedirs(D_R, exist_ok=True)
    temp_extract_dir = os.path.join(D_R, "temp_extract")
    
    try:
        # 清空临时目录（如果存在）
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

        # 解压 ZIP 到临时目录
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # 判断顶层目录结构
        top_level_items = os.listdir(temp_extract_dir)
        top_level_dirs = [d for d in top_level_items if os.path.isdir(os.path.join(temp_extract_dir, d))]

        # 选择 source_dir
        if len(top_level_dirs) == 1:
            source_dir = os.path.join(temp_extract_dir, top_level_dirs[0])
        else:
            source_dir = os.path.join(temp_extract_dir, model_type)

        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"未找到预期的目录：{source_dir}")

        # 移动文件（保留相对路径）
        for root, _, files in os.walk(source_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, source_dir)
                dst_path = os.path.join(D_R, rel_path)

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
        
        # 合并配置文件
        return True
        
    except Exception as e:
        return f"解压失败: {e}"
    finally:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
    
def process_configs(model_type,model_config,data_config):
    M_D = os.path.join(ROOT_PATH,'models')
    NEW_D =os.path.join(M_D,model_type)
    new_config = load_yaml_from_streamlit(model_config)
    MODEL_CONFIG_DIR =os.path.join(M_D,'models.yaml')
    OTHER_CONFIG_DIR = os.path.join(M_D,model_type,'models.yaml')
    os.makedirs(NEW_D, exist_ok=True)
    if os.path.exists(MODEL_CONFIG_DIR):
        with open(MODEL_CONFIG_DIR,'r') as f:
            old_config = yaml.safe_load(f) or {}
        merged = merge_model_configs(old_config,new_config)
        save_fixed_config_yaml(merged,MODEL_CONFIG_DIR)
        section = {k: v for k, v in merged.items() if k == model_type}
        if os.path.exists(OTHER_CONFIG_DIR):
            with open(OTHER_CONFIG_DIR,'r') as f:
                old_section_config = yaml.safe_load(f) or {}
                section_merged = merge_model_configs(old_section_config,section)
                save_fixed_config_yaml(section_merged,OTHER_CONFIG_DIR)
        else:
            save_fixed_config_yaml(section,OTHER_CONFIG_DIR)

    else:
        save_fixed_config_yaml(new_config,MODEL_CONFIG_DIR)
        save_fixed_config_yaml(new_config,OTHER_CONFIG_DIR)
    open_and_merge_data_yaml(model_type,data_config)
    
    

def process(model_type, model_zip, model_config, data_zip, data_config):
    processed_data = False
    processed_model = False
    res1, res2 = True,True
    if model_type != None and model_config!= None and data_config != None :
        process_configs(model_type,model_config,data_config)
        processed_data = True
        processed_model = True
    if model_type != None and data_config !=None and data_zip != None :
        res1=process_data(model_type,data_config,data_zip,processed_data)
        processed_data = True
    if model_type != None and data_config !=None and model_zip != None and model_config !=None :
        res2 = process_model(model_type,model_config,model_zip,data_config,processed_data,processed_model)
    if res1 == True and res2 == True:
        return True
    else:
        f"处理结果: 数据={res1}, 模型={res2}"

def delete(model_type, model_name):
    # 1. 读取 YAML 文件
    MODEL_CONFIG_PATH = os.path.join(ROOT_PATH, 'models', 'models.yaml')
    with open(MODEL_CONFIG_PATH, 'r') as f:
        model_config = yaml.safe_load(f)

    # 2. 删除对应的模型配置
    if model_type in model_config:
        model_section = model_config[model_type].get("models", {})
        if model_name in model_section:
            del model_section[model_name]
            # 如果 models 为空了，也可以选择一并删除（可选）
            if not model_section:
                del model_config[model_type]["models"]
        else:
            st.warning(f"未找到模型类型 {model_type} 下的模型 {model_name}")
    else:
        st.warning(f"未找到模型类型 {model_type}")

    # 3. 写回 YAML 文件
    save_fixed_config_yaml(model_config, MODEL_CONFIG_PATH)

    # 4. 删除对应的文件和目录
    MODEL_DIRECTORY_PATH = os.path.join(ROOT_PATH, 'models', model_type)

    def safe_remove(path):
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)  # 清除只读权限
            func(path)

        if os.path.isdir(path):
            shutil.rmtree(path, onerror=on_rm_error)
        elif os.path.isfile(path):
            os.remove(path)

    targets = [
        os.path.join(MODEL_DIRECTORY_PATH, model_name),
        os.path.join(MODEL_DIRECTORY_PATH, f"{model_name}_finetune"),
        os.path.join(MODEL_DIRECTORY_PATH, f"{model_name}_README.md")
    ]

    for path in targets:
        if os.path.exists(path):
            safe_remove(path)
        else:
            st.info(f"路径不存在：{path}")
    
    st.success(f"成功删除模型 {model_name} 在 {model_type} 下的所有相关内容。")

        
    
    


            


    
    


    
        
config = {
    # 分类任务数据集
    'Tox21': [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ],
    'ClinTox': ['FDA_APPROVED', 'CT_TOX'],
    'MUV': [
        'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
        'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
        'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
    ],
    'SIDER': [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
        'Product issues', 'Eye disorders', 'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances',
        'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders', 'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders',
        'Nervous system disorders', 'Injury, poisoning and procedural complications'
    ],
    'BBBP': ['p_np'],
    'HIV': ['HIV_active'],
    'BACE': ['Class'],
    
    # 回归任务数据集
    'FreeSolv': ['expt'],
    'ESOL': ['measured log solubility in mols per litre'],
    'Lipo': ['exp'],
    'qm7': ['u0_atom'],
    'qm8': [
        'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0',
        'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
    ],
    'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
}
dataset_names = ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
regression_tasks = ['FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']


def get_datasets():
    return dataset_names

def get_datasets_measure_names(dataset_name):
    """
    获取MoleculeNet数据集的目标列表。
    如果提供了dataset_name，则返回对应数据集的目标列表；
    否则返回包含所有数据集配置的字典。
    """
    # 如果指定了数据集名称，返回对应配置；否则返回完整配置
    try:
        return config[dataset_name]
    except KeyError:
        raise ValueError(f"无效的数据集名称: {dataset_name}, 获取对应属性失败\n支持的数据集名称: {dataset_names}")

def get_datasets_measure_numbers(dataset_name):
    return get_datasets_measure_names(dataset_name).__len__()

def get_datasets_task_type(dataset_name):
    return 'regression' if dataset_name in regression_tasks else 'classification'

def validate_datasets_measure_names(dataset_name, measure_name):
    if dataset_name not in dataset_names:
        raise ValueError(f"无效的数据集名称: {dataset_name}, 获取对应属性失败\n支持的数据集名称: {dataset_names}")
    if measure_name not in get_datasets_measure_names(dataset_name):
        raise ValueError(f"数据集{dataset_name}无效的目标名称: {measure_name}, 获取对应属性失败\n支持的目标名称: {get_datasets_measure_names(dataset_name)}")
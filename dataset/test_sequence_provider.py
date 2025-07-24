from base import BaseDataset

if __name__ == "__main__":
    dataset_name = "muv"
    csv_path = "../data/muv.csv"
    task_type = "classification"
    model_type = "molecule_net"

    # 初始化数据集
    ds = BaseDataset(datasetname=dataset_name, datasetpath=csv_path)
    ds.loadData()

    result = ds.get_smiles_and_labels_by_config("molecule_net", "classification")
    print("result = ds.get_smiles_and_labels_by_config('molecule_net', 'classification')")
    print(f"样本数 = {len(result['data'])}")
    print(f"SMILES: {result['data'][0]}")
    print(f"标签: {result['label'][0]}")
    print("=======================================")
    example_smiles = result["data"][0]
    data_val, label_val = ds.get_entry_by_smiles(
        smiles_str=example_smiles,
        target_col="MUV-466",
        model_type="molecule_net",
        task_type="classification"
    )
    data_col = "data"
    target_col = "MUV-466"
    print(f"SMILES: {example_smiles}")
    print(f"列: {data_col}): {data_val}")
    print(f"列: {target_col}): {label_val}")
    print("=======================================")

    all_info = ds.get_all_smiles_and_task_labels(model_type=model_type)
    print(f": {len(all_info['data'])}")
    print(f"所有任务和标签列:")
    for task, labels in all_info["tasks"].items():
        print(f"  - {task}: {labels}")

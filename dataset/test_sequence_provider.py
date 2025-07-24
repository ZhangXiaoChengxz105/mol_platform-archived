from base import BaseDataset

if __name__ == "__main__":
    dataset_name = "MUV"
    csv_path = "../data/muv.csv"
    config_file = "./data/moleculenet/dataset.yaml"


    ds = BaseDataset(datasetname=dataset_name, datasetpath=csv_path)
    ds.loadData()


    result = ds.get_data_and_labels_by_config(
        config_file=config_file
    )
    print("✅ 调用 get_data_and_labels_by_config 成功")
    print(f"样本数: {len(result['data'])}")
    print(f"🧪 示例数据: {result['data'][0]}")
    print(f"🏷️ 示例标签: {result['label'][0]}")
    print("=======================================")


    example_data = result["data"][0]
    data_val, label_val = ds.get_entry_by_data(
        data_str=example_data,
        target_col="MUV-466",
        config_file=config_file
    )
    print("✅ 调用 get_entry_by_data 成功")
    print(f"🔍 查找数据: {example_data}")
    print(f"📦 字段 data: {data_val}")
    print(f"🎯 字段 MUV-466: {label_val}")
    print("=======================================")

    all_info = ds.get_all_data_and_task_labels(
        config_file=config_file
    )
    print("✅ 调用 get_all_data_and_task_labels 成功")
    print(f"有效数据个数: {len(all_info['data'])}")
    print("📚 所有任务和标签列:")
    for task, labels in all_info["tasks"].items():
        print(f"  - {task}: {labels}")

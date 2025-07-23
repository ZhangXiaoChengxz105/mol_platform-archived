# Upload models rule
## Formatted workflow requirements
    To make sure the core file can be shown on platform
    you must have 3 core files: *_data.py, *_model.py, *_output.py
    which follows following structure

    model_name/
    ├── model_name_data.py      # convert the data from dataset to format that model can accept
    ├── model_name_model.py     # model core implementation
    ├── model_name_output.py    # formalize the model output

    This benefits other users to use your model
    by understanding the workflow of your model
    

## Least requirements (for not formatted workflow)
    To upload your customized model,
    no matter how many files the model needs,
    you must have a file named:
    model_output.py
    which should have a function named "predict" that:

    return a list results[] corresponding to input datas
    
    where elements in results are a dictionary result:
    {
        "data": smiles,
        "task": task,
        "prediction": prediction,
        "label": label,
    }


## structure (for coding convenience)
root/
	models/
	├── dataset_name1/
    │   ├── model_name1/            # workflow files
    │   ├── model_name1_finetune/   # pretrained parameters
    |   ├── model_name1_README.md   # model information
    │   ├── model_name2/
    │   ├── model_name2_finetune/
    |   ├── model_name2_README.md
    |   ...
    |   ├── model_nameN/
    |   ├── model_nameN_finetune/
    |   ├── model_nameN_README.md
    |   ...
    |   ├── model_datasets.yaml # 数据集信息 
    ...
    ├── dataset_nameN/
    |   ...
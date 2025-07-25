# **Models upload rule**
## **Formatted workflow requirements**
To fit the structure we define
you must have 3 core files: *_data.py, *_model.py, *_output.py, and paremeter files
which follows following structure ("model_name" is the name of model you set)

    model.zip/
    ├── model_name/                  # model workflow folder
    |    ├── model_name_data.py      # convert the data from dataset to format that model can accept (ex: smiles to fingerprints)
    |    ├── model_name_model.py     # model core implementation (ex: fingerprints to predict values)
    |    ├── model_name_output.py    # formalize the model output (ex: pred values to formatted results)
    |    ├── ...
    ├── model_name_finetune/         # pretrained parameters folder
    |    ├── ...
    ├── model_name_README.md         # model description, include environment requirement and usage

an example structure from moleculenet:

    fp/
    ├── fp_data.py
    ├── fp_model.py
    ├── fp_output.py
    ├── fp_test.py      # dev test file
    ├── pubchemfp.py    # other related files
    fp_finetune/
    ├── ...
    fp_README.md

and config files:

    model.yaml              # model config
    
    dataset.yaml            # dataset info


This benefits other users to use your model
by understanding the workflow of your model

The format is shown as below:

## **Least requirements (also for not formatted workflow)**
You can also upload your customized model,
but at lease have 3 files named:

    model_name_output.py    # replace with the "model_name" you set
    
    model.yaml
    
    dataset.yaml

### **model_name_output.py**
This file should have a function named "predict":
    
    def predict(name, target, data_list, model_type = None):  

        return results

parameters meaning:
ex: (moleculenet datasets, FP_NN model (which accept fingerprints as input, converted from smiles by FP_data.py))

    data_list = [smile1,smile2]     # data_type of moleculnet dataset
    name = "BBBP"                   # dataset (subdataset) name
    target = "p_np"                 # target (task) name
    model_type = 'NN'               # model type (model name) to select model for prediction
**although only one model in model_name_model.py, you should specify and use it, see below in model.yaml**

    results = predict(name, target, data_list, model_type = 'NN') # usage


results is a list of result,

    results = [result1, result2, ...]

result (element) in results is dictionary:

    result = {
        "data": smiles,
        "task": task,
        "prediction": prediction,
        "label": label,
    }

and a file named:

### **model.yaml**
This file should be formatted as below example from moleculnet:

    moleculenet:
    datasets: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

    models:
        # model_type
        FP:     # FP refer to fingerprint, which is the input of models (NN, RF,...)
            # model_name
            NN: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']    # supported datasets
            RF: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
            SVM: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
            XGB: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
        GNN:
            GIN: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
            GCN: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

        
        RNN:
            # although only one model in RNN, you should still specify it to show supported datasets
            RNN: ['ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
        SEQ:
            SEQ: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

### **dataset.yaml**
This file should be formatted as below example from moleculnet:
(you can only add the datasets and task you want to predict)

    data_type: "smiles"
    dataset_names: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
    regression_datasets: ['FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

    config: {
        # 分类任务数据集
        'Tox21': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
        ...,
        # 回归任务数据集
        'FreeSolv': ['expt'],
        ...
    }

## **detailed structure (for coding convenience)**
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
        ...
        ├── dataset_nameN/
        |   ...
        ├── model.yaml     # dataset info 
        ├── models_README.md        # intro



# **Models upload rule**

## **Least requirements (also for not formatted workflow)**
You can upload your customized model,
but at lease have 3 files named:

`model_name_output.py`    # replace with the "model_name" you set
    
`model.yaml`

`dataset.yaml`

`env.md`    # environment requirement instruction and environment dependancies (install -r requirements.txt)

`workflow_requirements.txt`

## **model_name_output.py**
This file should have a function named "predict":
    
    def predict(name, target, data_list, model_type = None):  

        return results

**parameters meaning:**
ex: (moleculenet datasets, FP_NN model (which accept fingerprints as input, converted from smiles by FP_data.py))

--`data_list`:  list of datas with data_type of moleculnet dataset
--`name`:       dataset (subdataset) name
--`target`:     target (task) name
--`model_type`: model type (model name) to select model for prediction, default to None for single model case

    data_list = [smile1,smile2]
    name` = "BBBP" 
    target = "p_np"
    model_type = 'NN'

    results = predict(name, target, data_list, model_type = 'NN') # usage

**return requirement:**
results returned is a list of result,

    results = [result1, result2, ...]

result (element) in results is dictionary:

    result = {
        "data": smiles,
        "task": task,
        "prediction": prediction,
        "label": label,
    }


## **model.yaml**
This file should be formatted as below example from moleculnet:
    # datasets name
    moleculenet:
        # subdatasets
        datasets: ['Tox21', 'ClinTox', 'MUV', 'SIDER', 'BBBP', 'HIV', 'BACE', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']
        
        models:
            # model_type, the input_type of models, ex: FP refers to fingerprint
            FP:
                # model_name, also as paremeter "model_type" in function predict
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

## **dataset.yaml**
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

# **Formatted structure (for coding convenience)**
## **Formatted workflow is recommended (but optional)**

this is the recommended structure of the folder after extraction

    dataset_type/                    # This must correspond to the 数据集类型 field you selected or specified on the website.
    ├── model_name/                  # model workflow folder
    |    ├── model_name_data.py      # convert the data from dataset to format that model can accept (ex: smiles to fingerprints)
    |    ├── model_name_model.py     # model core implementation (ex: fingerprints to predict values)
    |    ├── model_name_output.py    # formalize the model output (ex: pred values to formatted results)
    |    ├── ...
    ├── model_name_finetune/         # pretrained parameters folder
    |    ├── ...
    ├── model_name_README.md         # model description, include environment requirement and usage

an example workflow from moleculenet:

    data.py: smiles -> fingerprints
    model.py: fingerprints -> predict values
    output.py: predict values -> results

    smiles -> fingerprints -> predict values -> results
    
    
    
workflow structure:

    FP/                 # refers to fingerprints, which will be used in FP_model (coverted from smiles data by FP_data.py)
    ├── FP_data.py
    ├── FP_model.py
    ├── FP_output.py
    ├── FP_test.py      # dev test file
    ├── pubchemfp.py    # other related files
    FP_finetune/
    ├── ...
    FP_README.md

and config files:

    model.yaml              # model config
    
    dataset.yaml            # dataset info


This benefits other users to use your model
by understanding the workflow of your model




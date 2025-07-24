# Upload models rule
## Formatted workflow requirements
To fit the structure we define
you must have 3 core files: *_data.py, *_model.py, *_output.py
which follows following structure

    model_name/
    ├── model_name_data.py      # convert the data from dataset to format that model can accept (ex: smiles to fingerprints)
    ├── model_name_model.py     # model core implementation (ex: fingerprints to predict values)
    ├── model_name_output.py    # formalize the model output (ex: pred values to formatted results)

This benefits other users to use your model
by understanding the workflow of your model

the model_name output.py is required as follows:
        

### Least requirements (also for not formatted workflow)
You can also upload your customized model,
but at lease have a file named:

    model_output.py

which should have a function named "predict":
    
    def predict(name, target, smiles_list, model_type = None):

        return results
    
parameters meaning:
ex: (moleculenet datasets, FP_NN model (which accept fingerprints as input, converted from smiles by FP_data.py))

    smiles_list = [smile1,smile2]   # data_type of moleculnet dataset
    name = "BBBP"                   # sub-dataset name
    target = "p_np"                 # task name
    results = predict(name, target, smiles_list, model_type = 'NN') # usage

results is a list of result,

    results = [result1, result2, ...]

result (element) in results is dictionary:

    result = {
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
        ...
        ├── dataset_nameN/
        |   ...
        ├── model_datasets.yaml     # dataset info 
        ├── models_README.md        # intro
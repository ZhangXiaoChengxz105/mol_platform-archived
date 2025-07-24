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

which should have a function named "predict":
    
    def predict(name, target, smiles_list, model_type = None):

        return results
    
    '''
    results is a list of result,
    where result (element) in results is dictionary:
    result = {
        "data": smiles,
        "task": task,
        "prediction": prediction,
        "label": label,
    }
    '''

ex: (moleculenet datasets, FP_NN model)

    smiles_list = [smile1,smile2]
    name = "BBBP"       # sub-dataset name
    target = "p_np"     # task name
    results = predict(name, target, smiles_list, model_type = 'NN')

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
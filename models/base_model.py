class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.task = task(name)
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass

def task(name):
    # 根据属性自动决定task
    if name in ['BACE', 'BBBP', 'ClinTox', 'HIV', 'MUV', 'SIDER', 'Tox21']:
        return 'classification'
    else: return'regression'
from check_utils import get_datasets_task_type
class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.task = get_datasets_task_type(name)
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass
from models import base_model
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FingerprintModel(base_model):
    def __init__(self, name, path, fingerprint_type='morgan', fingerprint_size=2048):
        super().__init__(name, path)
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.model = None

    def featurize(self, smiles_list):
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if self.fingerprint_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fingerprint_size)
            elif self.fingerprint_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError("Unsupported fingerprint type")
            features.append(np.array(fp))
        return np.array(features)

    def load_weights(self, path):
        import joblib
        self.model = joblib.load(path)

    def train(self, X, y):
        if self.model is None:
            self.model = RandomForestClassifier()
        self.model.fit(X, y)
        # 可选：保存模型权重
        import joblib
        joblib.dump(self.model, f"{self.path}/fingerprint_model.joblib")

    def predict(self, data):
        features = self.featurize(data)
        return self.model.predict_proba(features)[:, 1]  # 假设二分类

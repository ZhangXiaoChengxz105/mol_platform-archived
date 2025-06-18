from base import BaseDataset
from provider import dataProvider
# from torch_geometric.data import Data
import pandas as pd

class graphDataset(BaseDataset, dataProvider):
    def loadData(self):
        df = pd.read_csv(self.datasetPath)
        self.data = df

    def preprocessData(self):
        pass

    def provideData(self, params):
        task_name = params.get("task_name", None)
        if not task_name:
            raise ValueError("pls define task_name")

        if task_name == 'BACE':
            smiles_col = 'mol'
        else:
            smiles_col = 'smiles'

        label_map = {
            'BBBP': ['p_np'],
            'Tox21': [
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
            ],
            'ClinTox': ['CT_TOX', 'FDA_APPROVED'],
            'HIV': ['HIV_active'],
            'BACE': ['Class'],
            'SIDER': [
                "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions", "Endocrine disorders",
                "Surgical and medical procedures", "Vascular disorders",
                "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders", "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders",
                "Injury, poisoning and procedural complications"
            ],
            'MUV': [
                'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858',
                'MUV-713', 'MUV-733', 'MUV-652', 'MUV-466', 'MUV-832'
            ],
            'FreeSolv': ['expt'],
            'ESOL': ['measured log solubility in mols per litre'],
            'Lipo': ['exp'],
            'qm7': ['u0_atom'],
            'qm8': [
                "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
                "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
            ],
            'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
        }

        if task_name not in label_map:
            raise ValueError(f"no task_name: {task_name}")

        label_cols = label_map[task_name]

        data_list = []
        for _, row in self.data.iterrows():
            smiles = row[smiles_col]
            label = row[label_cols[0]] if len(label_cols) == 1 else tuple(row[label_cols])
            data_list.append((smiles, label))

        return data_list

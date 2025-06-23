import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from finetune_pubchem_light import LightningModule as RM                    # regression model
from finetune_pubchem_light_classification import LightningModule as CM     # classification model
from finetune_pubchem_light_classification_multitask import MultitaskModel as MM    # multitask_classification model

from tokenizer.tokenizer import MolTranBertTokenizer
from argparse import Namespace
from fast_transformers.masking import LengthMask as LM

from check_utils import get_datasets_measure_names
from base_model import base_model
'''
class base_model:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.task = task(name)
    def load_weights(self, path):
        pass
    def predict(self, data):
        pass
'''
# models/SEQ/SEQ_model.py
from SEQ_data import TOKENIZER
class SEQ(base_model):
    def __init__(self, name, path):
        super().__init__(name, path)
        # SEQ细分task
        if self.name in ["Tox21","ClinTox","MUV","SIDER"]:
            self.task = "classification_multitask"
        # 初始化tokenizer
        self.tokenizer = TOKENIZER
        self.model = None
        
    def load_weights(self, path=None):
        load_path = path if path is not None else self.path
        if not load_path:
            raise ValueError("必须提供模型路径（path参数或初始化时的path）")
        
        config = self.configs(self.task, dataset_name=self.name).config
        config = Namespace(**config)
        self.model = self.get_model(config)
        
        if load_path.endswith(".pth"):
            try:
                self.model.load_state_dict(torch.load(load_path))
                self.model.eval()
                self.model.net.eval()
                print("load pth success!")
            except Exception as e:
                raise RuntimeError(f"权重加载失败: {str(e)}")
        elif load_path.endswith(".ckpt"):
            try:
                # 初始化模型（参数与pubchem完全一致）
                checkpoint = torch.load(load_path, map_location='cpu')  # 加载到 CPU
                model_state_dict = checkpoint.get('state_dict', checkpoint)  # 兼容不同格式
                # 移除可能存在的 'model.' 前缀（如果 .ckpt 文件保存时包含了此前缀）
                if all(key.startswith('model.') for key in model_state_dict.keys()):
                    model_state_dict = {k[6:]: v for k, v in model_state_dict.items()}
                self.model.load_state_dict(model_state_dict)
                self.model.eval()
                self.model.net.eval()
                print("load ckpt success!\n\n")
            except Exception as e:
                raise RuntimeError(f"权重加载失败: {str(e)}")
        else:
            print("invalid path!")

    def predict(self, data):
        if not self.model:
            raise RuntimeError("请先调用load_weights()")
        # 模拟 training_step 和 validation_step 中的处理流程
        idx = data[0]  # 假设第一个张量是输入的 token ids
        mask = data[1]
        b, t = idx.size()
        token_embeddings = self.model.tok_emb(idx)  # 词嵌入
        x = self.model.drop(token_embeddings)
        x = self.model.blocks(x, length_mask=LM(mask.sum(-1)))  # Transformer 块
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        smiles_emb = sum_embeddings / sum_mask
        self.model.eval()
        self.model.net.eval()
        with torch.no_grad():
            return self.model.net(smiles_emb)

    def get_model(self, config):
        if self.task == "classification_multitask":
            print("\nChoose multitask-classification model\n\n")
            return MM(config, self.tokenizer)
        
        elif self.task == "classification":
            print("\nchoose classification model\n\n")
            return CM(config, self.tokenizer)
        else:
            print("\nChoose regression model\n\n")
            return RM(config, self.tokenizer) 

    class configs:
        """提供完整默认值的配置类"""
        
        # 命令行参数完整默认值（与get_parser完全一致）
        CMD_DEFAULTS = {
            # Model参数
            'n_head': 12,
            'fold': 0,
            'n_layer': 12,
            'd_dropout': 0.1,
            'n_embd': 768,
            'fc_h': 512,
            'num_tasks': None,
            # Train参数
            'n_batch': 512,
            'from_scratch': False,
            'checkpoint_every': 1000,
            'lr_start': 3e-5,
            'lr_multiplier': 1,
            'n_jobs': 1,
            'device': 'cuda',
            'seed': 12345,
            'seed_path': "",
            'num_feats': 32,
            'max_epochs': 500,
            'seed_path': "../data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt",
            
            # 推理参数
            'mode': 'avg',  # 保留默认值，但训练脚本未显式设置
            'train_dataset_length': None,
            'eval_dataset_length': None,
            'desc_skip_connection': False,
            'num_workers': 8,
            'dropout': 0.1,
            'dims': [768, 768, 768, 1],
            'smiles_embedding': "/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/embeddings/protein/ba_embeddings_tanh_512_2986138_2.pt",
            'aug': None,
            'num_classes': None,
            'dataset_name': None,
            'measure_name': "measure",
            'checkpoints_folder': "models/SEQ_finetune",  # 必填项
            'checkpoint_root': None,
            'data_root': "/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/affinity",
            'batch_size': 128,
            
        }
        
        # 任务特定默认值（覆盖命令行默认值）
        TASK_DEFAULTS = {
            'classification': {
                'batch_size': 128,
                'num_classes': 2,
                # 'mode': 'cls',  # 训练脚本未显式设置，依赖模型内部默认值
            },
            'regression': {
                'batch_size': 128,
                # 'mode': 'reg',  # 训练脚本未显式设置
            },
            'classification_multitask': {
                'batch_size': 32,
                # 'mode': 'multitask_cls',  # 训练脚本未显式设置
            }
        }

        def __init__(self, task_type=None, **kwargs):
            # 初始化基础配置（命令行默认值）
            self.config = self.CMD_DEFAULTS.copy()
            # 合并任务特定配置
            if task_type in self.TASK_DEFAULTS:
                self.config.update(self.TASK_DEFAULTS[task_type])
            # 解析命令行风格参数
            self._parse_cmd_params(kwargs)
            
            # 用户参数（hparams）覆盖所有配置
            self.config.update(kwargs)
            
            # 校验必填参数
            if not self.config.get('checkpoints_folder'):
                raise ValueError("checkpoints_folder 为必填参数")
            
            # 设置 measure_names 和 num_tasks
            self._set_measure_name()

        def _parse_cmd_params(self, kwargs):
            """解析命令行风格的参数格式"""
            # 处理dims参数（字符串转列表）
            if 'dims' in kwargs and kwargs['dims'] == "[]":
                kwargs['dims'] = []
            elif 'dims' in kwargs and isinstance(kwargs['dims'], str):
                kwargs['dims'] = [int(d) for d in kwargs['dims'].split()]
            
            # 处理布尔值参数（字符串转bool）
            bool_params = ['from_scratch', 'desc_skip_connection']
            for param in bool_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    kwargs[param] = kwargs[param].lower() in ['true', '1', 'yes']

        def _set_measure_name(self):
            dataset_name = self.config.get('dataset_name')
            try:
                if dataset_name in ["Tox21", "ClinTox", "MUV", "SIDER"]:
                    self.config['measure_names'] = get_datasets_measure_names(dataset_name)
                else:
                    self.config['measure_names'] = []
                self.config['num_tasks'] = len(self.config.get('measure_names', []))
            except ValueError as e:
                print(f"错误: {e}，请检查数据集名称是否正确。")



"""
Authors: Nicolas Raymond

Description: Stores constant indicating the paths leading to 
             important directories within the project or pre-trained
             models on HuggingFace.
"""

from os.path import dirname, join


# Project paths
CURRENT_PROJECT: str = dirname(dirname(__file__))
CHECKPOINTS: str = join(CURRENT_PROJECT, 'checkpoints')
EXPERIMENTS: str = join(CURRENT_PROJECT, 'experiments')
EXPERIMENTS_CACHE: str = join(EXPERIMENTS, 'cache')
DATA: str = join(CURRENT_PROJECT, 'data')
RECORDS: str = join(CURRENT_PROJECT, 'records')
TRAINED_MODELS: str = join(CURRENT_PROJECT, 'models')
SRC: str = join(CURRENT_PROJECT, 'src')
MODELS: str = join(SRC, 'models')
RAW_DATA: str = join(DATA, 'raw')
INTERIM_DATA: str = join(DATA, 'interim')
PROCESSED_DATA: str = join(DATA, 'processed')
PFMG_DATA: str = join(PROCESSED_DATA, 'pea_faba_medicago_grasspea')
REVERSE_COMPLEMENT_MAP: str = join(SRC, 'data', 'modules', 'reverse_complement_map')
ORTHOGROUP_SPLIT: str = join(RECORDS, 'OG-SPLIT')

# HuggingFace paths
DDNABERT_HF: str = 'Peltarion/dnabert-minilm-small'
DNABERT_HF: str = 'zhihan1996/DNA_bert'

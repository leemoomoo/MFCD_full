import os
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))

target = os.path.join(SCRIPTS_DIR, "run_fairwisa_compare.py")

lines = []
lines.append('# coding: utf-8')
lines.append('import os, sys, json, copy, random, argparse')
lines.append('import numpy as np')
lines.append('import pandas as pd')
lines.append('import torch')
lines.append('import torch.nn as nn')
lines.append('import torch.optim as optim')
lines.append('from torch.utils.data import DataLoader')
lines.append('from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error')
lines.append('from tqdm import tqdm')
lines.append('')
lines.append('for p in [r"' + ROOT.replace('\\', '\\\\') + '", os.path.join(r"' + ROOT.replace('\\', '\\\\') + '", "experiments", "ncdm"), os.path.dirname(__file__)]:')
lines.append('    if p not in sys.path: sys.path.insert(0, p)')
lines.append('')
lines.append('from FairWISAFluency import FairWISAFluency, LearnableMatrix')
lines.append('from experiments.ncdm.math1_dataset import build_math1_splits_from_folder')
lines.append('from experiments.ncdm.math2_dataset import build_math2_splits_from_folder')
lines.append('from experiments.ncdm.frcsub_dataset import build_frcsub_splits_from_folder')
lines.append('from experiments.ncdm.Assist17_dataset import build_assist17_splits_from_folder')
lines.append('from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder')
lines.append('from experiments.ncdm.junyi_dataset import build_junyi_splits_from_csv')
lines.append('')
lines.append("ALL_DATASETS = ['math1', 'math2', 'frcsub', 'assist17', 'assist09', 'junyi']")
lines.append('')
# Now copy remaining functions from the original... 
# Actually let me just write the full file content

content = chr(10).join(lines)
open(target, 'w', encoding='utf-8').write(content)
print('Written:', target)
print(len(lines), 'lines')
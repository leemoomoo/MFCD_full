# coding: utf-8
\"\"\"Full equation trace: g_ij (Eq.10), s_flu (Eq.11), delta_ij, o_logit, p (Eq.12).\"\"\"
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.ncdm.NCDM_Fluency import NCDMFluency
from assist09_dataset import build_assist09_splits_from_folder, load_item_q_info, build_maps_from_train
DATA_DIR = r"D:\12487\NCDM PLUS2\data\a0910"
CKPT = r"D:\12487\NCDM PLUS2\checkpoints\ncdm_fluency_a0910.pt"
OUT_DIR = r"D:\12487\NCDM PLUS2\outputs\case_study"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

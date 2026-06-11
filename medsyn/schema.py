"""Feature schema for the wide MIMIC-CXR table."""
import numpy as np

BINARY_COLS = [
    "gender_F",              # 1
    "dx_pneumonia",          # 2
    "dx_pneumothorax",       # 3
    "dx_chf",                # 4
    "dx_pleural_effusion",   # 5
    "dx_atelectasis",        # 6
    "is_intubated",          # 7
    "has_pacemaker",         # 8
    "on_ventilator",         # 9
]

CONTINUOUS_COLS: list[str] = [
    # 你的 5 个 continuous 列名
    "anchor_age",
    "spo2",
    "resp_rate",
    "wbc",
    "bnp"

]

N_BIN = len(BINARY_COLS)
N_CONT = len(CONTINUOUS_COLS)

# CONT_MEAN = ...
# CONT_STD  = ...
CONT_MEAN = np.array([60.0, 96.5, 20.0, 10.5, 450.0], dtype=np.float32)
CONT_STD  = np.array([18.0,  4.0,  6.0,  6.0, 600.0], dtype=np.float32)
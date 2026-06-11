"""Wide-table MIMIC-CXR dataset."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .schema import BINARY_COLS, CONTINUOUS_COLS, CONT_MEAN, CONT_STD


# =============================================================================
# Dataset
# =============================================================================
def cxr_path(root, subject_id, study_id, dicom_id):
    p_top = f"p{str(subject_id)[:2]}"
    return Path(root) / "mimic" / p_top / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def clean_text(text, max_chars=4000):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\*\*[^\]]*\*\*\]", "[DEID]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:max_chars]


class WideTripleDataset(Dataset):
    def __init__(self, parquet_path, cxr_root, tokenizer,
                 image_size=224, max_note_tokens=256,
                 split=None, train=True):
        df = pd.read_parquet(parquet_path)
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.cxr_root = cxr_root
        self.tok = tokenizer
        self.max_note_tokens = max_note_tokens

        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- CXR ----
        if "cxr_path" in row and isinstance(row["cxr_path"], str):
            cp = row["cxr_path"]
        else:
            cp = str(cxr_path(self.cxr_root, int(row.subject_id),
                              int(row.study_id), row.dicom_id))
        cxr = self.tf(Image.open(cp).convert("RGB"))

        # ---- Tabular EHR ----
        binary = np.array([float(row.get(c, 0) or 0) for c in BINARY_COLS],
                          dtype=np.float32)
        cont_raw = np.array([row.get(c, np.nan) for c in CONTINUOUS_COLS],
                            dtype=np.float32)
        cont_mask = np.isfinite(cont_raw) # Mask of valid continuous features
        cont_z = np.where(cont_mask,
                          (cont_raw - CONT_MEAN) / CONT_STD, 0.0).astype(np.float32)

        # ---- Note (prefer impression) ----
        imp = clean_text(row.get("impression", ""))
        fnd = clean_text(row.get("findings", ""))
        text = imp or fnd or "[NO REPORT]"
        enc = self.tok(text, padding="max_length", truncation=True,
                       max_length=self.max_note_tokens, return_tensors="pt")

        return {
            "cxr":            cxr,
            "ehr_binary":     torch.from_numpy(binary),
            "ehr_cont":       torch.from_numpy(cont_z),
            "ehr_cont_mask":  torch.from_numpy(cont_mask),
            "note_input_ids": enc.input_ids[0],
            "note_attn_mask": enc.attention_mask[0].long(),
        }


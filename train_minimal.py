 """Minimal multi-modal synthetic patient training — single file, no packages.

What this is:
  * Reads the wide CXR feature parquet (from Data.ipynb)
  * Encodes (CXR + tabular EHR + radiology note) → 512-dim patient latent z
  * Decodes z back to (small CXR, EHR, note logits) — reconstruction loss
  * Single training loop, no DP, no causal DAG, no diffusion, no RoentGen,
    no Meditron, no BiomedCLIP

How to run:
  pip install torch torchvision timm pandas pillow transformers pyarrow
  python train_minimal.py \\
      --features /path/to/mimic_cxr_features.parquet \\
      --cxr-root /path/to/mimic-cxr-jpg-2.1.0.physionet.org \\
      --batch-size 8 --max-steps 200

Memory: ~3GB GPU, ~4GB CPU. Anything since 2018 runs it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Feature schema — must match Data.ipynb cell 26 output
# =============================================================================
BINARY_COLS = [
    "gender_F",
    "dx_pneumonia", "dx_pneumothorax", "dx_chf",
    "dx_pleural_effusion", "dx_atelectasis",
    "is_intubated", "has_pacemaker", "on_ventilator",
]
CONTINUOUS_COLS = ["anchor_age", "spo2", "resp_rate", "wbc", "bnp"]
N_BIN = len(BINARY_COLS)        # 9
N_CONT = len(CONTINUOUS_COLS)   # 5

# Rough z-score stats (override on your own training set if you want)
CONT_MEAN = np.array([60.0, 96.5, 20.0, 10.5, 450.0], dtype=np.float32)
CONT_STD  = np.array([18.0,  4.0,  6.0,  6.0, 600.0], dtype=np.float32)


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
        cont_mask = np.isfinite(cont_raw)
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


# =============================================================================
# Model — Patient encoder + three simple decoders
# =============================================================================
class PatientEncoder(nn.Module):
    """ViT (CXR) + MLP (tabular EHR) + ClinicalBERT (note) → 512-dim z."""

    def __init__(self, image_size=224, latent_dim=512, pretrained=True):
        super().__init__()
        import timm
        # CXR backbone
        self.cxr = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained,
            num_classes=0, img_size=image_size,
        )
        d_cxr = self.cxr.num_features                      # 768

        # Tabular EHR encoder
        d_ehr = 384
        in_dim = N_BIN + N_CONT + N_CONT
        self.ehr = nn.Sequential(
            nn.Linear(in_dim, 2 * d_ehr), nn.GELU(), nn.LayerNorm(2 * d_ehr),
            nn.Linear(2 * d_ehr, d_ehr), nn.LayerNorm(d_ehr),
        )

        # Note encoder (ClinicalBERT, ~110M params, fast, no dtype headaches)
        self.note = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        d_note = self.note.config.hidden_size              # 768

        # Cross-modal fusion via [PATIENT] token attention
        d_fuse = max(d_cxr, d_ehr, d_note)
        self.proj_cxr  = nn.Linear(d_cxr,  d_fuse)
        self.proj_ehr  = nn.Linear(d_ehr,  d_fuse)
        self.proj_note = nn.Linear(d_note, d_fuse)
        self.patient_tok = nn.Parameter(torch.zeros(1, 1, d_fuse))
        nn.init.trunc_normal_(self.patient_tok, std=0.02)

        layer = nn.TransformerEncoderLayer(d_fuse, 8, 4 * d_fuse,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d_fuse)
        self.to_latent = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.GELU(),
            nn.Linear(d_fuse, latent_dim),
        )

    def forward(self, batch):
        # CXR tokens
        cxr_h = self.cxr.forward_features(batch["cxr"])    # (B, 197, 768)
        if cxr_h.dim() == 2:
            cxr_h = cxr_h.unsqueeze(1)
        cxr_tok = self.proj_cxr(cxr_h)

        # EHR token (single)
        ehr_h = self.ehr(torch.cat([
            batch["ehr_binary"], batch["ehr_cont"],
            batch["ehr_cont_mask"].float(),
        ], dim=-1))
        ehr_tok = self.proj_ehr(ehr_h).unsqueeze(1)

        # Note tokens
        note_h = self.note(
            input_ids=batch["note_input_ids"],
            attention_mask=batch["note_attn_mask"],
        ).last_hidden_state
        note_tok = self.proj_note(note_h)

        # Concat with learnable [PATIENT] token in front
        B = cxr_tok.size(0)
        pt = self.patient_tok.expand(B, -1, -1)
        x = torch.cat([pt, cxr_tok, ehr_tok, note_tok], dim=1)

        # Build key-padding mask (only notes have padding)
        pad = torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)
        n_note = note_tok.size(1)
        pad[:, -n_note:] = ~batch["note_attn_mask"].bool()

        h = self.fusion(x, src_key_padding_mask=pad)
        return self.to_latent(self.norm(h[:, 0]))          # (B, latent_dim)


class CXRDecoder(nn.Module):
    """z → low-res CXR (64×64) for reconstruction loss. NOT photo-realistic;
    just a sanity head. Replace with diffusion when ready."""
    def __init__(self, latent_dim=512, out_size=64):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.GELU(),
            nn.Linear(1024, 3 * out_size * out_size),
        )

    def forward(self, z):
        return self.net(z).view(-1, 3, self.out_size, self.out_size)

    def loss(self, real_cxr, z):
        target = F.adaptive_avg_pool2d(real_cxr, self.out_size)
        return F.mse_loss(self.forward(z), target)


class TabularEHRDecoder(nn.Module):
    """z → (9 binary logits, 5 continuous values, 5 missingness logits)."""
    def __init__(self, latent_dim=512, d_model=384):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, 2 * d_model), nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head_binary = nn.Linear(d_model, N_BIN)
        self.head_cont   = nn.Linear(d_model, N_CONT)
        self.head_miss   = nn.Linear(d_model, N_CONT)

    def loss(self, binary, cont, mask, z):
        h = self.body(z)
        l_bin  = F.binary_cross_entropy_with_logits(self.head_binary(h), binary)
        mf = mask.float()
        cont_diff = (self.head_cont(h) - cont) ** 2
        l_cont = (cont_diff * mf).sum() / mf.sum().clamp(min=1.0)
        l_miss = F.binary_cross_entropy_with_logits(self.head_miss(h), mf)
        return l_bin + l_cont + 0.1 * l_miss


class NotesDecoder(nn.Module):
    """Small in-house causal LM (4 layers, 384 d_model) conditioned on z.
    NOT a state-of-the-art generator; just enough to give note reconstruction
    a meaningful gradient signal. Swap for Meditron later if needed."""
    def __init__(self, vocab_size, latent_dim=512, d_model=384,
                 n_layers=4, n_heads=6, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.cond = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight                # weight-tied
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len + 1, max_len + 1, dtype=torch.bool),
                       diagonal=1),
            persistent=False,
        )

    def forward(self, input_ids, attention_mask, z):
        B, T = input_ids.shape
        pos = torch.arange(T + 1, device=z.device)[None].expand(B, -1)
        x = torch.cat([
            self.cond(z).unsqueeze(1),
            self.emb(input_ids),
        ], dim=1) + self.pos(pos)
        mask = self.causal_mask[:T + 1, :T + 1]
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=z.device),
            ~attention_mask.bool(),
        ], dim=1)
        h = self.body(x, mask=mask, src_key_padding_mask=kpm)
        return self.head(h[:, 1:])                        # (B, T, V)

    def loss(self, input_ids, attention_mask, z):
        logits = self.forward(input_ids, attention_mask, z)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        # predict t+1 from t
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )


# =============================================================================
# Training loop
# =============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    print("loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    print("building dataset...", flush=True)
    ds = WideTripleDataset(
        args.features, args.cxr_root, tokenizer=tok,
        image_size=args.image_size, max_note_tokens=args.max_note_tokens,
        split=args.split, train=True,
    )
    print(f"  rows = {len(ds)}", flush=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True, pin_memory=True,
    )

    print("building models...", flush=True)
    enc = PatientEncoder(image_size=args.image_size,
                         latent_dim=args.latent_dim).to(device)
    cxr_dec = CXRDecoder(latent_dim=args.latent_dim).to(device)
    ehr_dec = TabularEHRDecoder(latent_dim=args.latent_dim).to(device)
    note_dec = NotesDecoder(
        vocab_size=tok.vocab_size, latent_dim=args.latent_dim,
        max_len=args.max_note_tokens,
    ).to(device)

    n_params = sum(p.numel() for m in [enc, cxr_dec, ehr_dec, note_dec]
                   for p in m.parameters())
    print(f"  total params = {n_params / 1e6:.1f}M", flush=True)

    optim = torch.optim.AdamW(
        list(enc.parameters()) + list(cxr_dec.parameters()) +
        list(ehr_dec.parameters()) + list(note_dec.parameters()),
        lr=args.lr,
    )

    for m in [enc, cxr_dec, ehr_dec, note_dec]:
        m.train()

    for step, batch in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        z = enc(batch)
        l_cxr  = cxr_dec.loss(batch["cxr"], z)
        l_ehr  = ehr_dec.loss(batch["ehr_binary"], batch["ehr_cont"],
                              batch["ehr_cont_mask"], z)
        l_note = note_dec.loss(batch["note_input_ids"],
                               batch["note_attn_mask"], z)
        loss = l_cxr + l_ehr + l_note

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        optim.step()

        if step % args.log_every == 0:
            print(f"step {step:5d}  loss={loss.item():7.4f}  "
                  f"cxr={l_cxr.item():6.4f}  ehr={l_ehr.item():6.4f}  "
                  f"note={l_note.item():7.4f}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder":  enc.state_dict(),
        "cxr_dec":  cxr_dec.state_dict(),
        "ehr_dec":  ehr_dec.state_dict(),
        "note_dec": note_dec.state_dict(),
        "args":     vars(args),
    }, out / "ckpt.pt")
    print(f"saved → {out / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="mimic_cxr_features.parquet from Data.ipynb")
    ap.add_argument("--cxr-root", required=True,
                    help="MIMIC-CXR-JPG root dir")
    ap.add_argument("--output",  default="runs/minimal")
    ap.add_argument("--split",   default=None,
                    help="train / val / test (only if parquet has split col)")
    ap.add_argument("--image-size",      type=int, default=224)
    ap.add_argument("--max-note-tokens", type=int, default=256)
    ap.add_argument("--latent-dim",      type=int, default=512)
    ap.add_argument("--batch-size",      type=int, default=8)
    ap.add_argument("--num-workers",     type=int, default=4)
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--max-steps",       type=int, default=None)
    ap.add_argument("--log-every",       type=int, default=10)
    train(ap.parse_args())

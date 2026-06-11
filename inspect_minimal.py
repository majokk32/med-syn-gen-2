from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer
 
# Make the project root importable so `from medsyn.X import ...` works whether
# you run this file directly or as `python -m inspect`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
from medsyn.data import WideTripleDataset
from medsyn.encoder import PatientEncoder
from medsyn.decoder import CXRDecoder, TabularEHRDecoder, NotesDecoder
from medsyn.schema import (
    BINARY_COLS, CONTINUOUS_COLS,
    CONT_MEAN, CONT_STD,
    N_BIN, N_CONT,
)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _denorm_image(t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized tensor (3, H, W) → uint8 (H, W, 3)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t.detach().cpu().float() * std + mean
    t = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return t
 
 
def _to_numpy(x):
    """torch tensor → numpy with float32, ensuring float for arithmetic."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to ckpt.pt saved by train.py")
    ap.add_argument("--features", required=True,
                    help="mimic_cxr_features.parquet from Data.ipynb")
    ap.add_argument("--cxr-root", required=True,
                    help="MIMIC-CXR-JPG root dir")
    ap.add_argument("--n", type=int, default=5,
                    help="how many patients to inspect")
    ap.add_argument("--split", default=None,
                    help="train/val/test (only if parquet has split col)")
    ap.add_argument("--out", default="runs/medsyn/inspect")
    ap.add_argument("--max-note-tokens", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--latent-dim", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="how many tokens to greedily decode for note")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device = {device}", flush=True)
 
    # ---- 1. ckpt ----
    print(f"loading ckpt: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    for k in ("encoder", "cxr_dec", "ehr_dec", "note_dec"):
        if k not in ckpt:
            raise KeyError(f"ckpt is missing key {k!r} — expected dict with "
                           f"keys encoder/cxr_dec/ehr_dec/note_dec")
 
    # ---- 2. tokenizer + dataset ----
    print("loading tokenizer + dataset (eval mode, no aug)...", flush=True)
    tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ds = WideTripleDataset(
        args.features, args.cxr_root, tokenizer=tok,
        image_size=args.image_size, max_note_tokens=args.max_note_tokens,
        split=args.split, train=False,
    )
    print(f"  rows = {len(ds)}", flush=True)
 
    # ---- 3. build models and load weights ----
    print("building models + loading weights...", flush=True)
    enc = PatientEncoder(image_size=args.image_size,
                         latent_dim=args.latent_dim).to(device)
    cxr_dec = CXRDecoder(latent_dim=args.latent_dim).to(device)
    ehr_dec = TabularEHRDecoder(latent_dim=args.latent_dim).to(device)
    note_dec = NotesDecoder(vocab_size=tok.vocab_size,
                             latent_dim=args.latent_dim,
                             max_len=args.max_note_tokens).to(device)
 
    enc.load_state_dict(ckpt["encoder"])
    cxr_dec.load_state_dict(ckpt["cxr_dec"])
    ehr_dec.load_state_dict(ckpt["ehr_dec"])
    note_dec.load_state_dict(ckpt["note_dec"])
 
    for m in (enc, cxr_dec, ehr_dec, note_dec):
        m.eval()
 
    # ---- 4. pick N samples ----
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)),
                      replace=False).tolist()
    print(f"inspecting samples: {idxs}", flush=True)
 
    md = [
        "# medsyn — Inspection",
        "",
        f"checkpoint: `{args.ckpt}`",
        f"samples: {idxs}",
        "",
    ]
 
    # ---- 5. per-sample reconstruction ----
    for k, idx in enumerate(idxs):
        sample = ds[idx]
        row = ds.df.iloc[idx]
        batch = {kk: vv.unsqueeze(0).to(device) for kk, vv in sample.items()}
 
        with torch.no_grad():
            z = enc(batch)
 
            # -- CXR reconstruction --
            cxr_recon = cxr_dec(z)                       # (1, 3, H', W')
 
            # -- Tabular reconstruction --
            # NOTE: this assumes TabularEHRDecoder exposes
            #       .body, .head_binary, .head_cont, .head_miss
            # If you renamed these, update the four lines below.
            h = ehr_dec.body(z)
            bin_prob = torch.sigmoid(ehr_dec.head_binary(h)).cpu().numpy()[0]
            cont_pred_z = ehr_dec.head_cont(h).cpu().numpy()[0]
            miss_prob = torch.sigmoid(ehr_dec.head_miss(h)).cpu().numpy()[0]
 
            # -- Note greedy decoding from z --
            bos = tok.cls_token_id or tok.bos_token_id or 101
            generated = [bos]
            for _ in range(args.gen_tokens):
                inp = torch.tensor([generated], device=device)
                attn = torch.ones_like(inp)
                logits = note_dec(inp, attn, z)
                nxt = int(logits[0, -1].argmax().item())
                generated.append(nxt)
                stop = tok.sep_token_id or tok.eos_token_id
                if stop is not None and nxt == stop:
                    break
            decoded = tok.decode(generated, skip_special_tokens=True)
 
        # ---- save CXR side-by-side ----
        real_img = _denorm_image(sample["cxr"])
        recon_img = _denorm_image(cxr_recon[0])
        recon_pil = Image.fromarray(recon_img).resize(
            (real_img.shape[1], real_img.shape[0]), Image.NEAREST,
        )
        side = np.concatenate([real_img, np.array(recon_pil)], axis=1)
        Image.fromarray(side).save(out / f"{k:02d}_cxr.png")
 
        # ---- write tabular comparison ----
        md.append(f"\n## Sample {k} (idx={idx})\n")
        md.append(f"subject_id={int(row['subject_id'])}, "
                  f"study_id={int(row['study_id'])}, "
                  f"dicom_id={str(row['dicom_id'])[:12]}...\n")
 
        md.append("### Binary flags  (real | pred prob)\n")
        md.append("| field | real | pred |")
        md.append("|---|---|---|")
        for i, col in enumerate(BINARY_COLS):
            real_v = int(row.get(col, 0) or 0)
            md.append(f"| {col} | {real_v} | {bin_prob[i]:.3f} |")
 
        md.append("\n### Continuous (real | pred | P(present))\n")
        md.append("| field | real | pred | P(present) |")
        md.append("|---|---|---|---|")
        for i, col in enumerate(CONTINUOUS_COLS):
            raw = row.get(col, np.nan)
            pred_clin = float(cont_pred_z[i] * CONT_STD[i] + CONT_MEAN[i])
            real_s = f"{raw:.2f}" if pd.notna(raw) else "NaN"
            md.append(f"| {col} | {real_s} | {pred_clin:.2f} "
                      f"| {miss_prob[i]:.3f} |")
 
        # ---- save full note files ----
        (out / f"{k:02d}_note_real.txt").write_text(
            str(row.get("impression", "") or "")[:2000], encoding="utf-8")
        (out / f"{k:02d}_note_gen.txt").write_text(
            decoded, encoding="utf-8")
 
        md.append("\n### Note (greedy decoded from z)\n")
        md.append("**Real impression** (first 200 chars):\n")
        md.append("> " + str(row.get("impression", "") or "")[:200]
                       .replace("\n", " "))
        md.append("\n**Generated** (first 200 chars):\n")
        md.append("> " + decoded[:200].replace("\n", " "))
        md.append(f"\n**Image**: ![cxr]({k:02d}_cxr.png)\n")
 
    # ---- 6. write report ----
    (out / "inspect.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved → {out}/", flush=True)
    print(f"  inspect.md", flush=True)
    print(f"  {len(idxs)} × _cxr.png   (real | reconstructed)", flush=True)
    print(f"  {len(idxs)} × _note_real.txt, _note_gen.txt", flush=True)
 
 
if __name__ == "__main__":
    main()
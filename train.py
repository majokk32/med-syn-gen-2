"""Training entry — single script that wires everything together."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from medsyn.data import WideTripleDataset
from medsyn.encoder import PatientEncoder
from medsyn.decoder import CXRDecoder, TabularEHRDecoder, NotesDecoder

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
        loss = 3.0 * l_cxr + 1.0 * l_ehr + 1.0 * l_note   

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
    ap.add_argument("--max-steps",       type=int, default=4000)
    ap.add_argument("--log-every",       type=int, default=10)
    train(ap.parse_args())
"""S2 adapter training with frozen MedVAE and legacy multimodal fusion."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from medsyn.medvae_fusion import (
    MedVAEFusionAutoencoder,
    load_legacy_multimodal_weights,
    trainable_parameters,
)
from medsyn.spatial_cxr import (
    ResizeAndPad,
    reconstruction_metrics,
)
from medsyn.spatial_multimodal import SubjectSplitTripleDataset


class MedVAETripleDataset(SubjectSplitTripleDataset):
    """Full-CXR padded input for MedVAE plus the existing EHR/report fields."""

    def __init__(self, *args, image_size: int = 512, **kwargs):
        super().__init__(*args, image_size=image_size, **kwargs)
        self.tf = transforms.Compose([
            ResizeAndPad(image_size),
            transforms.ToTensor(),
        ])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def compute_losses(model, batch, args):
    output = model(batch)
    z_global = output["z_global"]
    ehr = model.ehr_decoder.loss(
        batch["ehr_binary"],
        batch["ehr_cont"],
        batch["ehr_cont_mask"],
        z_global,
    )
    note = model.note_decoder.loss(
        batch["note_input_ids"],
        batch["note_attn_mask"],
        z_global,
    )
    cxr_summary = F.smooth_l1_loss(
        output["cxr_summary_pred"],
        output["cxr_summary_target"],
    )
    total = (
        args.ehr_weight * ehr
        + args.note_weight * note
        + args.cxr_summary_weight * cxr_summary
    )
    return total, {
        "loss": total,
        "ehr": ehr,
        "note": note,
        "cxr_summary": cxr_summary,
        "z_global_std": z_global.std(),
        "z_spatial_std": output["z_spatial"].std(),
    }, output


@torch.inference_mode()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {
        "loss": 0.0,
        "ehr": 0.0,
        "note": 0.0,
        "cxr_summary": 0.0,
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    seen = 0
    shape_info = None
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        batch = move_batch(batch, device)
        loss, stats, output = compute_losses(model, batch, args)
        metrics = reconstruction_metrics(
            output["cxr_reconstruction"], batch["cxr"]
        )
        n = batch["cxr"].size(0)
        values = {
            "loss": loss,
            "ehr": stats["ehr"],
            "note": stats["note"],
            "cxr_summary": stats["cxr_summary"],
            **metrics,
        }
        for key in totals:
            totals[key] += float(values[key]) * n
        seen += n
        if shape_info is None:
            shape_info = {
                "image": list(batch["cxr"].shape),
                "z_spatial": list(output["z_spatial"].shape),
                "cxr_embedding": list(output["cxr_embedding"].shape),
                "z_global": list(output["z_global"].shape),
            }
    model.train()
    if not seen:
        raise RuntimeError("validation produced no samples")
    return {
        **{key: value / seen for key, value in totals.items()},
        "shapes": shape_info,
    }


def save_checkpoint(path, model, optimizer, step, best_val, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen MedVAE weights are intentionally excluded; reload official model.
    weights = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("medvae.")
    }
    torch.save({
        "model": weights,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "args": vars(args),
        "medvae_model": model.medvae_name,
    }, path)


def train(args):
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("S2 requires GPU unless --allow-cpu is set")

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"runs/medvae_fusion_s2_{stamp}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    train_ds = MedVAETripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=args.max_note_tokens,
        split="train",
        train=True,
        limit=args.limit_train_rows,
    )
    val_ds = MedVAETripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=args.max_note_tokens,
        split="val",
        train=False,
        limit=args.limit_val_rows,
    )
    train_loader = make_loader(
        train_ds, args, shuffle=True, drop_last=True
    )
    val_loader = make_loader(
        val_ds, args, shuffle=False, drop_last=False
    )
    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    model = MedVAEFusionAutoencoder(
        vocab_size=tokenizer.vocab_size,
        medvae_model=args.medvae_model,
        max_note_tokens=args.max_note_tokens,
        global_dim=args.global_dim,
    )
    legacy_state = load_legacy_multimodal_weights(
        model, args.multimodal_ckpt
    )
    model.freeze_pretrained_backbones()
    model.to(device)

    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in parameters)
    print(
        f"params total={total_params / 1e6:.1f}M "
        f"trainable={train_params / 1e6:.1f}M "
        f"legacy_step={legacy_state.get('step', 'unknown')}",
        flush=True,
    )

    initial = evaluate(model, val_loader, device, args)
    print(json.dumps({"initial_validation": initial}, indent=2),
          flush=True)
    if args.max_steps == 0:
        print("S2 integration smoke test: PASS", flush=True)
        return

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )
    iterator = iter(train_loader)
    best_val = float("inf")
    model.train()

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            loss, stats, _ = compute_losses(model, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            parameters, args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[s2] step={step:5d} "
                f"loss={float(stats['loss']):.4f} "
                f"ehr={float(stats['ehr']):.4f} "
                f"note={float(stats['note']):.4f} "
                f"cxr_summary={float(stats['cxr_summary']):.4f} "
                f"zg_std={float(stats['z_global_std']):.4f} "
                f"zs_std={float(stats['z_spatial_std']):.4f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.max_steps:
            val = evaluate(model, val_loader, device, args)
            print(
                f"[s2-val] step={step:5d} "
                f"loss={val['loss']:.4f} "
                f"ehr={val['ehr']:.4f} "
                f"note={val['note']:.4f} "
                f"cxr_summary={val['cxr_summary']:.4f} "
                f"psnr={val['psnr']:.2f} "
                f"ssim={val['ssim']:.4f}",
                flush=True,
            )
            save_checkpoint(
                output / "ckpt_last.pt",
                model, optimizer, step, best_val, args,
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(
                    output / "ckpt_best.pt",
                    model, optimizer, step, best_val, args,
                )
                print(
                    f"new best S2 validation loss={best_val:.4f}",
                    flush=True,
                )
    print(f"S2 checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--multimodal-ckpt", required=True)
    parser.add_argument("--output")
    parser.add_argument("--medvae-model", default="medvae_4_3_2d")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-note-tokens", type=int, default=256)
    parser.add_argument("--global-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ehr-weight", type=float, default=1.0)
    parser.add_argument("--note-weight", type=float, default=0.25)
    parser.add_argument("--cxr-summary-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    train(parser.parse_args())

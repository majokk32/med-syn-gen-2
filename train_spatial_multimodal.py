"""S1 training: three-modality global latent plus conditional spatial CXR."""

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
from transformers import AutoTokenizer

from medsyn.spatial_cxr import (
    image_gradient_loss,
    reconstruction_metrics,
    structural_similarity,
)
from medsyn.spatial_multimodal import (
    SpatialMultimodalAutoencoder,
    SubjectSplitTripleDataset,
    checkpoint_exists,
    load_multimodal_checkpoint,
    load_s0_spatial_checkpoint,
    trainable_parameters,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, args, shuffle, drop_last):
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


def cxr_loss(recon, target, edge_weight, ssim_weight):
    l1 = F.l1_loss(recon, target)
    edge = image_gradient_loss(recon, target)
    ssim = structural_similarity(recon, target)
    total = l1 + edge_weight * edge + ssim_weight * (1.0 - ssim)
    return total, {"cxr_l1": l1, "cxr_edge": edge, "cxr_ssim": ssim}


def forward_losses(model, batch, args):
    recon, z_global, z_spatial, cxr_pixel = model.reconstruct_cxr(batch)
    l_cxr, image_stats = cxr_loss(
        recon, cxr_pixel, args.edge_weight, args.ssim_weight
    )
    l_ehr = model.ehr_decoder.loss(
        batch["ehr_binary"],
        batch["ehr_cont"],
        batch["ehr_cont_mask"],
        z_global,
    )
    l_note = model.note_decoder.loss(
        batch["note_input_ids"],
        batch["note_attn_mask"],
        z_global,
    )
    total = (
        args.cxr_weight * l_cxr
        + args.ehr_weight * l_ehr
        + args.note_weight * l_note
    )
    stats = {
        "loss": total,
        "cxr": l_cxr,
        "ehr": l_ehr,
        "note": l_note,
        "z_global_mean": z_global.mean(),
        "z_global_std": z_global.std(),
        "z_spatial_mean": z_spatial.mean(),
        "z_spatial_std": z_spatial.std(),
        **image_stats,
    }
    cache = {
        "recon": recon,
        "z_global": z_global,
        "z_spatial": z_spatial,
        "cxr_pixel": cxr_pixel,
    }
    return total, stats, cache


@torch.no_grad()
def evaluate(model, val_loader, device, args):
    model.eval()
    keys = [
        "loss", "cxr", "ehr", "note", "mae", "psnr", "ssim",
        "shuffle_mae_delta", "zero_mae_delta",
    ]
    totals = {key: 0.0 for key in keys}
    seen = 0

    for batch_index, batch in enumerate(val_loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        batch = move_batch(batch, device)
        total, stats, cache = forward_losses(model, batch, args)

        correct = cache["recon"]
        z_global = cache["z_global"]
        z_spatial = cache["z_spatial"]
        target = cache["cxr_pixel"]
        shuffled = model.cxr_decoder(
            z_spatial, torch.roll(z_global, shifts=1, dims=0)
        )
        zero = model.cxr_decoder(z_spatial, torch.zeros_like(z_global))

        metrics = reconstruction_metrics(correct, target)
        correct_mae = metrics["mae"]
        shuffle_delta = F.l1_loss(shuffled, target) - correct_mae
        zero_delta = F.l1_loss(zero, target) - correct_mae
        n = target.size(0)

        values = {
            "loss": total,
            "cxr": stats["cxr"],
            "ehr": stats["ehr"],
            "note": stats["note"],
            "mae": metrics["mae"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "shuffle_mae_delta": shuffle_delta,
            "zero_mae_delta": zero_delta,
        }
        for key, value in values.items():
            totals[key] += float(value) * n
        seen += n

    model.set_stage1_train_mode()
    if seen == 0:
        raise RuntimeError("validation loader produced no samples")
    return {key: value / seen for key, value in totals.items()}


def save_checkpoint(
    path, model, optimizer, step, best_joint, best_cxr, args
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_joint": best_joint,
        "best_cxr": best_cxr,
        "args": vars(args),
    }, path)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("S1 requires a GPU; pass --allow-cpu only for smoke")

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"runs/spatial_multimodal_s1_{stamp}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"device = {device}", flush=True)
    print(f"output = {output}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    train_ds = SubjectSplitTripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=args.max_note_tokens,
        split="train",
        train=True,
        limit=args.limit_train_rows,
    )
    val_ds = SubjectSplitTripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=args.max_note_tokens,
        split="val",
        train=False,
        limit=args.limit_val_rows,
    )
    print(f"subject split: train={len(train_ds)} val={len(val_ds)}",
          flush=True)
    train_loader = loader(train_ds, args, shuffle=True, drop_last=True)
    val_loader = loader(val_ds, args, shuffle=False, drop_last=False)

    model = SpatialMultimodalAutoencoder(
        vocab_size=tokenizer.vocab_size,
        image_size=args.image_size,
        max_note_tokens=args.max_note_tokens,
        global_dim=args.global_dim,
        spatial_channels=args.spatial_channels,
        spatial_base_channels=args.spatial_base_channels,
    )
    spatial_state = load_s0_spatial_checkpoint(model, args.spatial_ckpt)
    multimodal_state = load_multimodal_checkpoint(
        model, args.multimodal_ckpt
    )
    model.freeze_stage1_pretrained_parts(
        train_global_backbones=args.train_global_backbones,
        train_spatial_base=args.train_spatial_base,
    )
    model.to(device)

    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    train_params = sum(parameter.numel() for parameter in parameters)
    print(
        f"params total={total_params / 1e6:.1f}M "
        f"trainable={train_params / 1e6:.1f}M",
        flush=True,
    )
    print(
        f"loaded S0 step={spatial_state.get('step')} and "
        f"multimodal step={multimodal_state.get('step', 'unknown')}",
        flush=True,
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    iterator = iter(train_loader)
    best_joint = float("inf")
    best_cxr = float("inf")
    model.set_stage1_train_mode()

    initial = evaluate(model, val_loader, device, args)
    print(
        f"[s1-val] step={0:5d} loss={initial['loss']:.4f} "
        f"cxr={initial['cxr']:.4f} ehr={initial['ehr']:.4f} "
        f"note={initial['note']:.4f} mae={initial['mae']:.4f} "
        f"psnr={initial['psnr']:.2f} ssim={initial['ssim']:.4f} "
        f"shuffle_delta={initial['shuffle_mae_delta']:+.6f} "
        f"zero_delta={initial['zero_mae_delta']:+.6f}",
        flush=True,
    )

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
            loss, stats, _ = forward_losses(model, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[s1] step={step:5d} loss={float(stats['loss']):.4f} "
                f"cxr={float(stats['cxr']):.4f} "
                f"ehr={float(stats['ehr']):.4f} "
                f"note={float(stats['note']):.4f} "
                f"ssim={float(stats['cxr_ssim']):.4f} "
                f"zg_std={float(stats['z_global_std']):.4f} "
                f"zs_std={float(stats['z_spatial_std']):.4f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.max_steps:
            val = evaluate(model, val_loader, device, args)
            print(
                f"[s1-val] step={step:5d} loss={val['loss']:.4f} "
                f"cxr={val['cxr']:.4f} ehr={val['ehr']:.4f} "
                f"note={val['note']:.4f} mae={val['mae']:.4f} "
                f"psnr={val['psnr']:.2f} ssim={val['ssim']:.4f} "
                f"shuffle_delta={val['shuffle_mae_delta']:+.6f} "
                f"zero_delta={val['zero_mae_delta']:+.6f}",
                flush=True,
            )
            save_checkpoint(
                output / "ckpt_last.pt",
                model, optimizer, step, best_joint, best_cxr, args,
            )
            if val["loss"] < best_joint:
                best_joint = val["loss"]
                save_checkpoint(
                    output / "ckpt_best_joint.pt",
                    model, optimizer, step, best_joint, best_cxr, args,
                )
                print(f"new best S1 joint loss={best_joint:.4f}",
                      flush=True)
            if val["cxr"] < best_cxr:
                best_cxr = val["cxr"]
                save_checkpoint(
                    output / "ckpt_best_cxr.pt",
                    model, optimizer, step, best_joint, best_cxr, args,
                )
                print(f"new best S1 CXR loss={best_cxr:.4f}",
                      flush=True)

    print(f"S1 checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--spatial-ckpt", required=True,
                        type=checkpoint_exists)
    parser.add_argument("--multimodal-ckpt", required=True,
                        type=checkpoint_exists)
    parser.add_argument("--output")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-note-tokens", type=int, default=256)
    parser.add_argument("--global-dim", type=int, default=512)
    parser.add_argument("--spatial-channels", type=int, default=4)
    parser.add_argument("--spatial-base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cxr-weight", type=float, default=3.0)
    parser.add_argument("--ehr-weight", type=float, default=1.0)
    parser.add_argument("--note-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument("--ssim-weight", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--max-val-batches", type=int, default=10)
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-global-backbones", action="store_true")
    parser.add_argument("--train-spatial-base", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    train(parser.parse_args())

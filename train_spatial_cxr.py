"""Train the isolated spatial-latent CXR reconstruction experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medsyn.spatial_cxr import (
    SpatialCXRAutoencoder,
    SpatialCXRDataset,
    reconstruction_metrics,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, num_workers, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model, loader, device, max_batches, edge_weight, ssim_weight
):
    model.eval()
    totals = {"loss": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    seen = 0

    for batch_index, image in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        image = image.to(device, non_blocking=True)
        recon, latent = model(image)
        loss, _ = model.reconstruction_loss(
            image, recon, latent, edge_weight, ssim_weight
        )
        metrics = reconstruction_metrics(recon, image)
        n = image.size(0)
        totals["loss"] += float(loss) * n
        for key in ("mae", "psnr", "ssim"):
            totals[key] += float(metrics[key]) * n
        seen += n

    model.train()
    if seen == 0:
        raise RuntimeError("validation loader produced no samples")
    return {key: value / seen for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, step, best_val, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "args": vars(args),
    }, path)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"device = {device}", flush=True)

    train_ds = SpatialCXRDataset(
        args.features,
        args.cxr_root,
        image_size=args.image_size,
        split="train",
        train=True,
        limit=args.limit_train_rows,
    )
    val_ds = SpatialCXRDataset(
        args.features,
        args.cxr_root,
        image_size=args.image_size,
        split="val",
        train=False,
        limit=args.limit_val_rows,
    )
    print(
        f"subject-level split rows: train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    train_loader = make_loader(
        train_ds, args.batch_size, args.num_workers, True, True
    )
    val_loader = make_loader(
        val_ds, args.batch_size, args.num_workers, False, False
    )

    model = SpatialCXRAutoencoder(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model params = {param_count / 1e6:.2f}M", flush=True)
    print(
        f"spatial latent = ({args.latent_channels}, "
        f"{args.image_size // 8}, {args.image_size // 8})",
        flush=True,
    )

    start_step = 0
    best_val = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state["best_val"])
        print(f"resumed {args.resume} at step {start_step}", flush=True)

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    iterator = iter(train_loader)
    model.train()

    for step in range(start_step + 1, args.max_steps + 1):
        try:
            image = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image = next(iterator)
        image = image.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, stats = model.loss(
                image,
                edge_weight=args.edge_weight,
                ssim_weight=args.ssim_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[spatial-cxr] step={step:5d} "
                f"loss={float(stats['loss']):.4f} "
                f"l1={float(stats['l1']):.4f} "
                f"edge={float(stats['edge']):.4f} "
                f"ssim={float(stats['ssim']):.4f} "
                f"z_mean={float(stats['latent_mean']):.4f} "
                f"z_std={float(stats['latent_std']):.4f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.max_steps:
            val = evaluate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                args.edge_weight,
                args.ssim_weight,
            )
            print(
                f"[validation] step={step:5d} "
                f"loss={val['loss']:.4f} mae={val['mae']:.4f} "
                f"psnr={val['psnr']:.2f} ssim={val['ssim']:.4f}",
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
                    f"new best validation loss: {best_val:.4f}",
                    flush=True,
                )

    print(f"checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--output", default="runs/spatial_cxr")
    parser.add_argument("--resume")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument("--ssim-weight", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    train(parser.parse_args())

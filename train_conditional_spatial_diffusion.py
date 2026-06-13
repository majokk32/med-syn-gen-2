"""S3: train p(z_spatial | z_global) with frozen S2 and MedVAE."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from medsyn.conditional_diffusion import (
    ChannelNormalizer,
    ConditionalLatentDiffusion,
    ConditionalSpatialUNet,
)
from medsyn.medvae_fusion import (
    MedVAEFusionAutoencoder,
    load_s2_checkpoint,
)
from medsyn.spatial_cxr import ResizeAndPad
from medsyn.spatial_multimodal import SubjectSplitTripleDataset


class DiffusionTripleDataset(SubjectSplitTripleDataset):
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


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def make_loader(dataset, args, shuffle: bool, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def build_s2(args, tokenizer, device):
    state = torch.load(args.s2_ckpt, map_location="cpu")
    config = state.get("args", {})
    medvae_name = state.get(
        "medvae_model", config.get("medvae_model", "medvae_4_3_2d")
    )
    max_note_tokens = int(config.get(
        "max_note_tokens", args.max_note_tokens
    ))
    global_dim = int(config.get("global_dim", args.global_dim))
    model = MedVAEFusionAutoencoder(
        vocab_size=tokenizer.vocab_size,
        medvae_model=medvae_name,
        max_note_tokens=max_note_tokens,
        global_dim=global_dim,
    )
    load_s2_checkpoint(model, args.s2_ckpt)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, medvae_name, max_note_tokens, global_dim


@torch.inference_mode()
def encode_targets(model, batch):
    z_spatial = model.encode_spatial(batch["cxr"])
    z_global, _ = model.global_encoder(batch, z_spatial)
    return z_global, z_spatial


@torch.inference_mode()
def estimate_latent_statistics(model, loader, device, max_batches):
    sum_channels = None
    square_sum_channels = None
    count = 0
    latent_shape = None
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        _, latent = encode_targets(model, batch)
        latent = latent.float()
        reduced = latent.sum(dim=(0, 2, 3))
        square_reduced = latent.square().sum(dim=(0, 2, 3))
        sum_channels = (
            reduced if sum_channels is None else sum_channels + reduced
        )
        square_sum_channels = (
            square_reduced
            if square_sum_channels is None
            else square_sum_channels + square_reduced
        )
        count += latent.size(0) * latent.size(2) * latent.size(3)
        latent_shape = tuple(latent.shape[1:])
    if not count or latent_shape is None:
        raise RuntimeError("could not estimate MedVAE latent statistics")
    mean = sum_channels / count
    variance = square_sum_channels / count - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return mean.cpu(), std.cpu(), latent_shape


def update_ema(ema_model, model, decay: float) -> None:
    with torch.no_grad():
        ema_parameters = dict(ema_model.named_parameters())
        for name, parameter in model.named_parameters():
            ema_parameters[name].mul_(decay).add_(
                parameter, alpha=1.0 - decay
            )
        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)


@torch.inference_mode()
def evaluate(
    diffusion,
    s2_model,
    normalizer,
    loader,
    device,
    max_batches,
    min_snr_gamma,
):
    diffusion.eval()
    total = 0.0
    seen = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(17_321)
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        z_global, z_spatial = encode_targets(s2_model, batch)
        normalized = normalizer.normalize(z_spatial.float())
        timesteps = torch.randint(
            0,
            diffusion.timesteps,
            (normalized.size(0),),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            normalized.shape,
            device=device,
            dtype=normalized.dtype,
            generator=generator,
        )
        loss, _ = diffusion.training_loss(
            normalized,
            z_global.float(),
            timesteps=timesteps,
            noise=noise,
            min_snr_gamma=min_snr_gamma,
        )
        total += float(loss) * normalized.size(0)
        seen += normalized.size(0)
    diffusion.train()
    if not seen:
        raise RuntimeError("validation produced no samples")
    return total / seen


def save_checkpoint(
    path,
    diffusion,
    ema_diffusion,
    optimizer,
    step,
    best_val,
    mean,
    std,
    latent_shape,
    args,
    s2_metadata,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "diffusion": diffusion.state_dict(),
        "ema_diffusion": ema_diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "latent_mean": mean,
        "latent_std": std,
        "latent_shape": latent_shape,
        "args": vars(args),
        "s2": s2_metadata,
    }, path)


def train(args):
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("S3 requires GPU unless --allow-cpu is set")

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"runs/conditional_spatial_diffusion_s3_{stamp}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    s2_model, medvae_name, max_note_tokens, global_dim = build_s2(
        args, tokenizer, device
    )
    train_ds = DiffusionTripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=max_note_tokens,
        split="train",
        train=True,
        limit=args.limit_train_rows,
    )
    val_ds = DiffusionTripleDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=max_note_tokens,
        split="val",
        train=False,
        limit=args.limit_val_rows,
    )
    train_loader = make_loader(
        train_ds, args, shuffle=True, drop_last=True
    )
    stats_loader = make_loader(
        train_ds, args, shuffle=False, drop_last=False
    )
    val_loader = make_loader(
        val_ds, args, shuffle=False, drop_last=False
    )
    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"medvae={medvae_name}",
        flush=True,
    )

    mean, std, latent_shape = estimate_latent_statistics(
        s2_model, stats_loader, device, args.stats_batches
    )
    normalizer = ChannelNormalizer(mean, std).to(device)
    print(
        f"latent_shape={latent_shape} "
        f"mean={mean.tolist()} std={std.tolist()}",
        flush=True,
    )

    denoiser = ConditionalSpatialUNet(
        latent_channels=latent_shape[0],
        condition_dim=global_dim,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        condition_dropout=args.condition_dropout,
    )
    diffusion = ConditionalLatentDiffusion(
        denoiser, timesteps=args.diffusion_steps
    ).to(device)
    ema_diffusion = copy.deepcopy(diffusion).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainable = sum(
        parameter.numel()
        for parameter in diffusion.parameters()
        if parameter.requires_grad
    )
    print(f"diffusion trainable params={trainable / 1e6:.2f}M", flush=True)

    initial_val = evaluate(
        diffusion, s2_model, normalizer, val_loader,
        device, args.max_val_batches, args.min_snr_gamma,
    )
    print(f"initial validation noise_mse={initial_val:.4f}", flush=True)
    if args.max_steps == 0:
        print("S3 conditional diffusion smoke test: PASS", flush=True)
        return

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    iterator = iter(train_loader)
    best_val = float("inf")
    diffusion.train()

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        with torch.inference_mode():
            z_global, z_spatial = encode_targets(s2_model, batch)
            clean = normalizer.normalize(z_spatial.float())
            condition = z_global.float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            loss, stats = diffusion.training_loss(
                clean,
                condition,
                min_snr_gamma=args.min_snr_gamma,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            diffusion.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_diffusion, diffusion, args.ema_decay)

        if step == 1 or step % args.log_every == 0:
            print(
                f"[s3] step={step:6d} "
                f"noise_mse={float(stats['noise_mse']):.4f} "
                f"pred_std={float(stats['prediction_std']):.4f} "
                f"clean_std={float(stats['clean_std']):.4f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.max_steps:
            val = evaluate(
                ema_diffusion, s2_model, normalizer, val_loader,
                device, args.max_val_batches, args.min_snr_gamma,
            )
            print(
                f"[s3-val] step={step:6d} noise_mse={val:.4f}",
                flush=True,
            )
            metadata = {
                "checkpoint": args.s2_ckpt,
                "medvae_model": medvae_name,
                "global_dim": global_dim,
                "max_note_tokens": max_note_tokens,
            }
            save_checkpoint(
                output / "ckpt_last.pt",
                diffusion, ema_diffusion, optimizer,
                step, best_val, mean, std, latent_shape,
                args, metadata,
            )
            if val < best_val:
                best_val = val
                save_checkpoint(
                    output / "ckpt_best.pt",
                    diffusion, ema_diffusion, optimizer,
                    step, best_val, mean, std, latent_shape,
                    args, metadata,
                )
                print(
                    f"new best S3 validation noise_mse={best_val:.4f}",
                    flush=True,
                )
    print(f"S3 checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--s2-ckpt", required=True)
    parser.add_argument("--output")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-note-tokens", type=int, default=256)
    parser.add_argument("--global-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--channel-mults", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--stats-batches", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    train(parser.parse_args())

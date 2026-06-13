"""Generate CXR spatial latents from z_global and decode with frozen MedVAE."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
from medsyn.spatial_cxr import ResizeAndPad, reconstruction_metrics
from medsyn.spatial_multimodal import SubjectSplitTripleDataset


class InspectionDataset(SubjectSplitTripleDataset):
    def __init__(self, *args, image_size: int = 512, **kwargs):
        super().__init__(*args, image_size=image_size, **kwargs)
        self.tf = transforms.Compose([
            ResizeAndPad(image_size),
            transforms.ToTensor(),
        ])

    def local_index_for_source(self, source_index: int) -> int:
        matches = np.flatnonzero(
            self.df["_source_index"].to_numpy() == source_index
        )
        if len(matches) != 1:
            raise KeyError(
                f"source index {source_index} occurs {len(matches)} times"
            )
        return int(matches[0])


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().cpu().clamp(0, 1)
        .mul(255).byte().permute(1, 2, 0).numpy()
    )


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(120, 8 * len(label) + 12)
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def build_models(args, checkpoint, tokenizer, device):
    diffusion_args = checkpoint["args"]
    s2_meta = checkpoint["s2"]
    s2_model = MedVAEFusionAutoencoder(
        vocab_size=tokenizer.vocab_size,
        medvae_model=s2_meta["medvae_model"],
        max_note_tokens=int(s2_meta["max_note_tokens"]),
        global_dim=int(s2_meta["global_dim"]),
    )
    s2_path = args.s2_ckpt or s2_meta["checkpoint"]
    load_s2_checkpoint(s2_model, s2_path)
    s2_model.requires_grad_(False)
    s2_model.to(device).eval()

    latent_shape = tuple(checkpoint["latent_shape"])
    denoiser = ConditionalSpatialUNet(
        latent_channels=latent_shape[0],
        condition_dim=int(s2_meta["global_dim"]),
        base_channels=int(diffusion_args["base_channels"]),
        channel_mults=tuple(diffusion_args["channel_mults"]),
        condition_dropout=float(diffusion_args["condition_dropout"]),
    )
    diffusion = ConditionalLatentDiffusion(
        denoiser, timesteps=int(diffusion_args["diffusion_steps"])
    )
    weights = (
        checkpoint["ema_diffusion"]
        if args.use_ema else checkpoint["diffusion"]
    )
    diffusion.load_state_dict(weights)
    diffusion.to(device).eval()
    normalizer = ChannelNormalizer(
        checkpoint["latent_mean"], checkpoint["latent_std"]
    ).to(device)
    return s2_model, diffusion, normalizer, latent_shape, s2_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--s2-ckpt")
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--out")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="test")
    parser.add_argument("--source-indices")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--clip-denoised", type=float, default=5.0)
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    (
        s2_model,
        diffusion,
        normalizer,
        latent_shape,
        s2_path,
    ) = build_models(args, checkpoint, tokenizer, device)

    s2_meta = checkpoint["s2"]
    diffusion_args = checkpoint["args"]
    dataset = InspectionDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=int(diffusion_args["image_size"]),
        max_note_tokens=int(s2_meta["max_note_tokens"]),
        split=args.split,
        train=False,
    )
    if args.source_indices:
        source_indices = [
            int(value.strip())
            for value in args.source_indices.split(",")
            if value.strip()
        ]
        local_indices = [
            dataset.local_index_for_source(value)
            for value in source_indices
        ]
    else:
        rng = np.random.default_rng(args.seed)
        local_indices = rng.choice(
            len(dataset), size=min(args.n, len(dataset)), replace=False
        ).tolist()
        source_indices = [
            int(dataset.df.iloc[index]["_source_index"])
            for index in local_indices
        ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(
        args.out or f"runs/conditional_spatial_diffusion_inspect_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []

    for sample_number, local_index in enumerate(local_indices):
        sample = dataset[local_index]
        row = dataset.df.iloc[local_index]
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
        }
        with torch.inference_mode():
            real_spatial = s2_model.encode_spatial(batch["cxr"])
            z_global, _ = s2_model.global_encoder(batch, real_spatial)
            direct_reconstruction = s2_model.decode_spatial(real_spatial)
            initial_noise = torch.randn(
                (1, *latent_shape), device=device
            )
            generated_normalized = diffusion.sample(
                z_global.float(),
                latent_shape=latent_shape,
                steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                clip_denoised=args.clip_denoised,
                initial_noise=initial_noise,
            )
            generated = s2_model.decode_spatial(
                normalizer.denormalize(generated_normalized)
            )
            shuffled_condition = torch.zeros_like(z_global)
            shuffled_normalized = diffusion.sample(
                shuffled_condition.float(),
                latent_shape=latent_shape,
                steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                clip_denoised=args.clip_denoised,
                initial_noise=initial_noise,
            )
            shuffled = s2_model.decode_spatial(
                normalizer.denormalize(shuffled_normalized)
            )
            metrics = reconstruction_metrics(generated, batch["cxr"])
            direct_metrics = reconstruction_metrics(
                direct_reconstruction, batch["cxr"]
            )
            condition_delta = torch.mean(
                torch.abs(generated - shuffled)
            ).item()

        panel = np.concatenate([
            add_label(to_uint8(batch["cxr"][0]), "real"),
            add_label(
                to_uint8(direct_reconstruction[0]), "MedVAE recon"
            ),
            add_label(to_uint8(generated[0]), "z_global -> diffusion"),
            add_label(to_uint8(shuffled[0]), "zero condition"),
        ], axis=1)
        Image.fromarray(panel).save(
            output / f"{sample_number:02d}_cxr_panel.png"
        )
        rows.append({
            "sample": sample_number,
            "source_index": source_indices[sample_number],
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "mae": float(metrics["mae"]),
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "direct_psnr": float(direct_metrics["psnr"]),
            "direct_ssim": float(direct_metrics["ssim"]),
            "condition_delta": condition_delta,
        })

    summary = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "s2_checkpoint": s2_path,
        "split": args.split,
        "source_indices": source_indices,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "clip_denoised": args.clip_denoised,
        "latent_shape": latent_shape,
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "mean_direct_psnr": float(np.mean([
            row["direct_psnr"] for row in rows
        ])),
        "mean_direct_ssim": float(np.mean([
            row["direct_ssim"] for row in rows
        ])),
        "mean_condition_delta": float(np.mean([
            row["condition_delta"] for row in rows
        ])),
        "samples": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"saved S3 inspection to {output}")


if __name__ == "__main__":
    main()

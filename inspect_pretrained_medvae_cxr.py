"""Compare pretrained MedVAE CXR reconstruction with the local spatial AE."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from medsyn.spatial_cxr import (
    SpatialCXRAutoencoder,
    SpatialCXRDataset,
    reconstruction_metrics,
)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().cpu().clamp(0, 1)
        .mul(255).byte().permute(1, 2, 0).numpy()
    )


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(90, 8 * len(label) + 12)
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0))
    draw.text((5, 4), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def load_local_model(
    checkpoint: str, device: torch.device
) -> tuple[SpatialCXRAutoencoder, dict]:
    state = torch.load(checkpoint, map_location="cpu")
    config = state.get("args", {})
    model = SpatialCXRAutoencoder(
        latent_channels=int(config.get("latent_channels", 4)),
        base_channels=int(config.get("base_channels", 32)),
        upsample_mode=str(config.get("upsample_mode", "bilinear")),
    ).to(device)
    model.load_state_dict(state["model"])
    model.requires_grad_(False).eval()
    return model, state


def load_medvae(model_name: str, device: torch.device):
    try:
        from medvae import MVAE
    except ImportError as exc:
        raise SystemExit(
            "MedVAE is not installed. Install it in an isolated environment "
            "with `python -m pip install medvae`."
        ) from exc

    print(f"loading pretrained {model_name}...", flush=True)
    model = MVAE(model_name=model_name, modality="xray").to(device)
    model.requires_grad_(False).eval()
    return model


@torch.inference_mode()
def reconstruct_medvae(
    model, image: torch.Tensor, posterior: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # MedVAE is trained with images normalized to [-1, 1].
    normalized = image.mul(2.0).sub(1.0)
    latent_dist = model.model.encode(normalized)
    latent = (
        latent_dist.mode()
        if posterior == "mode"
        else latent_dist.sample()
    )
    recon = model.model.decode(latent).add(1.0).div(2.0).clamp(0, 1)
    return recon, latent


def metric_row(
    sample_number: int,
    row,
    model_name: str,
    recon: torch.Tensor,
    image: torch.Tensor,
    latent: torch.Tensor,
) -> dict:
    metrics = reconstruction_metrics(recon, image)
    return {
        "sample": sample_number,
        "source_index": int(row["_source_index"]),
        "subject_id": int(row["subject_id"]),
        "study_id": int(row["study_id"]),
        "dicom_id": str(row["dicom_id"]),
        "model": model_name,
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "latent_shape": "x".join(str(v) for v in latent.shape[1:]),
        "latent_mean": float(latent.mean()),
        "latent_std": float(latent.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--local-ckpt")
    parser.add_argument(
        "--medvae-models",
        default="medvae_8_4_2d,medvae_4_3_2d",
        help="comma-separated MedVAE model names",
    )
    parser.add_argument(
        "--posterior",
        choices=["mode", "sample"],
        default="mode",
        help="use posterior mode for deterministic reconstruction comparison",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test"
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-indices",
        help="comma-separated original parquet row indices",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    dataset = SpatialCXRDataset(
        args.features,
        args.cxr_root,
        image_size=args.image_size,
        split=None if args.split == "all" else args.split,
        train=False,
    )
    if args.source_indices:
        source_indices = [
            int(value.strip())
            for value in args.source_indices.split(",")
            if value.strip()
        ]
        indices = [
            dataset.local_index_for_source(value)
            for value in source_indices
        ]
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            len(dataset), size=min(args.n, len(dataset)), replace=False
        ).tolist()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(
        args.out or f"runs/pretrained_medvae_cxr_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=True)

    samples = []
    for sample_number, local_index in enumerate(indices):
        image = dataset[local_index].unsqueeze(0)
        samples.append({
            "sample": sample_number,
            "local_index": local_index,
            "row": dataset.df.iloc[local_index],
            "image": image,
            "recons": {},
        })

    rows = []
    local_state = None
    if args.local_ckpt:
        local_model, local_state = load_local_model(args.local_ckpt, device)
        for sample in samples:
            image = sample["image"].to(device)
            with torch.inference_mode():
                recon, latent = local_model(image)
            sample["recons"]["local_sharp"] = recon[0].cpu()
            rows.append(metric_row(
                sample["sample"], sample["row"], "local_sharp",
                recon, image, latent,
            ))
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_names = [
        value.strip()
        for value in args.medvae_models.split(",")
        if value.strip()
    ]
    for model_name in model_names:
        model = load_medvae(model_name, device)
        for sample in samples:
            image = sample["image"].to(device)
            recon, latent = reconstruct_medvae(
                model, image, args.posterior
            )
            sample["recons"][model_name] = recon[0].cpu()
            rows.append(metric_row(
                sample["sample"], sample["row"], model_name,
                recon, image, latent,
            ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    panel_order = (
        (["local_sharp"] if args.local_ckpt else []) + model_names
    )
    for sample in samples:
        real = sample["image"][0]
        panels = [add_label(to_uint8(real), "real")]
        for model_name in panel_order:
            panels.append(add_label(
                to_uint8(sample["recons"][model_name]), model_name
            ))
        Image.fromarray(np.concatenate(panels, axis=1)).save(
            output / f"{sample['sample']:02d}_comparison.png"
        )

    with (output / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for model_name in panel_order:
        selected = [row for row in rows if row["model"] == model_name]
        summaries[model_name] = {
            "samples": len(selected),
            "latent_shape": sorted({
                row["latent_shape"] for row in selected
            }),
            "mean_mae": float(np.mean([
                row["mae"] for row in selected
            ])),
            "mean_psnr": float(np.mean([
                row["psnr"] for row in selected
            ])),
            "mean_ssim": float(np.mean([
                row["ssim"] for row in selected
            ])),
        }
    summary = {
        "local_checkpoint": args.local_ckpt,
        "local_checkpoint_step": (
            local_state.get("step") if local_state else None
        ),
        "medvae_models": model_names,
        "posterior": args.posterior,
        "image_size": args.image_size,
        "split": args.split,
        "source_indices": [
            int(sample["row"]["_source_index"])
            for sample in samples
        ],
        "models": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved comparison to {output}", flush=True)


if __name__ == "__main__":
    main()

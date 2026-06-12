"""Inspect a spatial CXR checkpoint with real/reconstruction/error panels."""

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
    draw.rectangle((0, 0, 90, 22), fill=(0, 0, 0))
    draw.text((5, 4), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="test")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-indices",
        help="comma-separated original parquet row indices for fair comparison",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    config = state.get("args", {})
    image_size = int(config.get("image_size", 224))
    latent_channels = int(config.get("latent_channels", 4))
    base_channels = int(config.get("base_channels", 32))
    upsample_mode = str(config.get("upsample_mode", "bilinear"))

    model = SpatialCXRAutoencoder(
        latent_channels=latent_channels,
        base_channels=base_channels,
        upsample_mode=upsample_mode,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = SpatialCXRDataset(
        args.features,
        args.cxr_root,
        image_size=image_size,
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
    output = Path(args.out or f"runs/spatial_cxr/inspection/{timestamp}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for sample_number, local_index in enumerate(indices):
        image = dataset[local_index].unsqueeze(0).to(device)
        row = dataset.df.iloc[local_index]
        with torch.no_grad():
            recon, latent = model(image)
            metrics = reconstruction_metrics(recon, image)

        real_np = to_uint8(image[0])
        recon_np = to_uint8(recon[0])
        error_np = to_uint8((recon[0] - image[0]).abs().mul(3.0))
        panel = np.concatenate([
            add_label(real_np, "real"),
            add_label(recon_np, "recon"),
            add_label(error_np, "error x3"),
        ], axis=1)
        Image.fromarray(panel).save(
            output / f"{sample_number:02d}_cxr_panel.png"
        )

        rows.append({
            "sample": sample_number,
            "local_index": local_index,
            "source_index": int(row["_source_index"]),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "mae": float(metrics["mae"]),
            "mse": float(metrics["mse"]),
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "latent_mean": float(latent.mean()),
            "latent_std": float(latent.std()),
        })

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": args.ckpt,
        "checkpoint_step": state.get("step"),
        "split": args.split,
        "samples": len(rows),
        "source_indices": [row["source_index"] for row in rows],
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved inspection to {output}", flush=True)


if __name__ == "__main__":
    main()

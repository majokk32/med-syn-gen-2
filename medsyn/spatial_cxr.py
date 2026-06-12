"""Spatial-latent CXR autoencoder used by the isolated S0 experiment.

The baseline decoder reconstructs a 224x224 image from one global vector.
Here the image keeps a spatial bottleneck:

    CXR (3, 224, 224) -> z_spatial (C, 28, 28) -> CXR reconstruction

There are intentionally no encoder-to-decoder skip connections. Every image
detail must pass through ``z_spatial`` so the experiment measures whether a
spatial latent is a useful replacement for the baseline global-vector image
bottleneck.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .data import cxr_path


def _subject_bucket(subject_id: object, buckets: int = 100) -> int:
    """Stable hash bucket used for leakage-free subject-level splits."""
    digest = hashlib.blake2b(
        str(subject_id).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % buckets


class SpatialCXRDataset(Dataset):
    """Image-only view of the wide table with deterministic subject splits."""

    def __init__(
        self,
        parquet_path: str,
        cxr_root: str,
        image_size: int = 224,
        split: Optional[str] = None,
        train: bool = False,
        limit: Optional[int] = None,
        resize_mode: str = "crop",
    ):
        if resize_mode not in {"crop", "pad"}:
            raise ValueError("resize_mode must be crop or pad")
        df = pd.read_parquet(parquet_path).copy()
        df["_source_index"] = range(len(df))

        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be train, val, test, or None")
            bucket = df["subject_id"].map(_subject_bucket)
            keep = {
                "train": bucket < 80,
                "val": (bucket >= 80) & (bucket < 90),
                "test": bucket >= 90,
            }[split]
            df = df.loc[keep]

        if limit is not None:
            df = df.iloc[:limit]

        self.df = df.reset_index(drop=True)
        self.cxr_root = cxr_root

        if resize_mode == "pad":
            self.transform = transforms.Compose([
                ResizeAndPad(image_size),
                transforms.ToTensor(),
            ])
        elif train:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.05)),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.iloc[index]
        if "cxr_path" in row and isinstance(row["cxr_path"], str):
            path = Path(row["cxr_path"])
        else:
            path = cxr_path(
                self.cxr_root,
                int(row["subject_id"]),
                int(row["study_id"]),
                str(row["dicom_id"]),
            )
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def local_index_for_source(self, source_index: int) -> int:
        matches = self.df.index[self.df["_source_index"] == source_index]
        if len(matches) == 0:
            raise KeyError(
                f"source row {source_index} is not present in this split"
            )
        return int(matches[0])


class ResizeAndPad:
    """Fit the full image inside a square without changing aspect ratio."""

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        image = TF.resize(
            image,
            [resized_height, resized_width],
            antialias=True,
        )
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        right = self.size - resized_width - left
        bottom = self.size - resized_height - top
        return TF.pad(
            image,
            [left, top, right, bottom],
            fill=self.fill,
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpatialCXREncoder(nn.Module):
    """224x224 CXR to a 28x28 spatial latent."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, b, 5, stride=2, padding=2),       # 224 -> 112
            ResidualBlock(b),
            nn.Conv2d(b, 2 * b, 4, stride=2, padding=1),   # 112 -> 56
            ResidualBlock(2 * b),
            nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1),  # 56 -> 28
            ResidualBlock(4 * b),
            ResidualBlock(4 * b),
            nn.GroupNorm(8, 4 * b),
            nn.SiLU(),
            nn.Conv2d(4 * b, latent_channels, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class SpatialCXRDecoder(nn.Module):
    """28x28 spatial latent to a 224x224 CXR."""

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 32,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        if upsample_mode not in {"bilinear", "nearest"}:
            raise ValueError("upsample_mode must be bilinear or nearest")
        b = base_channels
        self.in_proj = nn.Conv2d(latent_channels, 4 * b, 3, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(4 * b),
            ResidualBlock(4 * b),
        )
        self.up1 = self._up_block(4 * b, 2 * b, upsample_mode)  # 28 -> 56
        self.up2 = self._up_block(2 * b, b, upsample_mode)      # 56 -> 112
        self.up3 = self._up_block(b, b, upsample_mode)          # 112 -> 224
        self.out = nn.Sequential(
            nn.GroupNorm(8, b),
            nn.SiLU(),
            nn.Conv2d(b, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _up_block(
        c_in: int, c_out: int, upsample_mode: str
    ) -> nn.Sequential:
        if upsample_mode == "bilinear":
            upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
        return nn.Sequential(
            upsample,
            nn.Conv2d(c_in, c_out, 3, padding=1),
            ResidualBlock(c_out),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.mid(self.in_proj(latent))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.out(x)


class SpatialCXRAutoencoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 32,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.encoder = SpatialCXREncoder(latent_channels, base_channels)
        self.decoder = SpatialCXRDecoder(
            latent_channels, base_channels, upsample_mode
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(image)
        return self.decode(latent), latent

    def loss(
        self,
        image: torch.Tensor,
        edge_weight: float = 0.10,
        ssim_weight: float = 0.20,
        laplacian_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon, latent = self(image)
        return self.reconstruction_loss(
            image,
            recon,
            latent,
            edge_weight,
            ssim_weight,
            laplacian_weight,
        )

    def reconstruction_loss(
        self,
        image: torch.Tensor,
        recon: torch.Tensor,
        latent: torch.Tensor,
        edge_weight: float = 0.10,
        ssim_weight: float = 0.20,
        laplacian_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = F.l1_loss(recon, image)
        edge = image_gradient_loss(recon, image)
        ssim = structural_similarity(recon, image)
        laplacian = (
            laplacian_pyramid_loss(recon, image)
            if laplacian_weight > 0
            else recon.new_zeros(())
        )
        total = (
            l1
            + edge_weight * edge
            + ssim_weight * (1.0 - ssim)
            + laplacian_weight * laplacian
        )
        stats = {
            "loss": total,
            "l1": l1,
            "edge": edge,
            "ssim": ssim,
            "laplacian": laplacian,
            "latent_mean": latent.mean(),
            "latent_std": latent.std(),
        }
        return total, stats


def image_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def laplacian_pyramid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    levels: int = 3,
) -> torch.Tensor:
    """Compare high-frequency residuals at several image scales."""
    total = pred.new_zeros(())
    pred_level = pred
    target_level = target
    for level in range(levels):
        pred_down = F.avg_pool2d(
            pred_level, kernel_size=2, stride=2, ceil_mode=True
        )
        target_down = F.avg_pool2d(
            target_level, kernel_size=2, stride=2, ceil_mode=True
        )
        pred_up = F.interpolate(
            pred_down, size=pred_level.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        target_up = F.interpolate(
            target_down, size=target_level.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        weight = 2.0 ** level
        total = total + weight * F.l1_loss(
            pred_level - pred_up, target_level - target_up
        )
        pred_level = pred_down
        target_level = target_down
    return total / sum(2.0 ** level for level in range(levels))


def structural_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: int = 11,
) -> torch.Tensor:
    """Dependency-free local SSIM for images in [0, 1]."""
    padding = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, 1, padding) - mu_y.square()
    sigma_x = sigma_x.clamp_min(0.0)
    sigma_y = sigma_y.clamp_min(0.0)
    sigma_xy = F.avg_pool2d(pred * target, window, 1, padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = (
        (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        / ((mu_x.square() + mu_y.square() + c1)
           * (sigma_x + sigma_y + c2))
    )
    return score.mean().clamp(-1.0, 1.0)


@torch.no_grad()
def reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    return {
        "mae": F.l1_loss(pred, target),
        "mse": mse,
        "psnr": -10.0 * torch.log10(mse.clamp_min(1e-10)),
        "ssim": structural_similarity(pred, target),
    }

"""S1 model: shared patient latent plus a conditional spatial CXR latent.

The shared patient latent keeps the existing three-modality path:

    normalized CXR + EHR + report -> PatientEncoder -> z_patient_global

The image-specific path keeps spatial detail:

    pixel CXR -> SpatialCXREncoder -> z_cxr_spatial

The CXR decoder receives both:

    CXR = decoder(z_cxr_spatial, condition=z_patient_global)

FiLM layers are zero-initialized, so loading an S0 spatial checkpoint starts
from the same reconstruction instead of immediately perturbing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .data import WideTripleDataset
from .decoder import NotesDecoder, TabularEHRDecoder
from .encoder import PatientEncoder
from .spatial_cxr import (
    SpatialCXRDecoder,
    SpatialCXREncoder,
    _subject_bucket,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_denormalize(image: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized CXR tensors back to [0, 1]."""
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


class SubjectSplitTripleDataset(WideTripleDataset):
    """Existing triple dataset with deterministic subject-level splitting."""

    def __init__(
        self,
        parquet_path,
        cxr_root,
        tokenizer,
        image_size=224,
        max_note_tokens=256,
        split: Optional[str] = None,
        train=True,
        limit: Optional[int] = None,
    ):
        super().__init__(
            parquet_path,
            cxr_root,
            tokenizer,
            image_size=image_size,
            max_note_tokens=max_note_tokens,
            split=None,
            train=train,
        )
        self.df["_source_index"] = range(len(self.df))
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be train, val, test, or None")
            bucket = self.df["subject_id"].map(_subject_bucket)
            keep = {
                "train": bucket < 80,
                "val": (bucket >= 80) & (bucket < 90),
                "test": bucket >= 90,
            }[split]
            self.df = self.df.loc[keep]
        if limit is not None:
            self.df = self.df.iloc[:limit]
        self.df = self.df.reset_index(drop=True)


class FiLM(nn.Module):
    """Feature-wise linear modulation from the shared patient latent."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self, feature: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return feature * (1.0 + scale) + shift


class ConditionalSpatialCXRDecoder(SpatialCXRDecoder):
    """S0 decoder augmented with zero-initialized global FiLM conditioning."""

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 32,
        condition_dim: int = 512,
    ):
        super().__init__(latent_channels, base_channels)
        b = base_channels
        self.film_mid = FiLM(condition_dim, 4 * b)
        self.film_up1 = FiLM(condition_dim, 2 * b)
        self.film_up2 = FiLM(condition_dim, b)
        self.film_up3 = FiLM(condition_dim, b)

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = self.mid(self.in_proj(latent))
        x = self.film_mid(x, condition)
        x = self.film_up1(self.up1(x), condition)
        x = self.film_up2(self.up2(x), condition)
        x = self.film_up3(self.up3(x), condition)
        return self.out(x)

    def film_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("film_"):
                yield parameter

    def freeze_pretrained_base(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.film_parameters():
            parameter.requires_grad_(True)


class SpatialMultimodalAutoencoder(nn.Module):
    """Three-modality shared latent with a separate spatial image latent."""

    def __init__(
        self,
        vocab_size: int,
        image_size: int = 224,
        max_note_tokens: int = 256,
        global_dim: int = 512,
        spatial_channels: int = 4,
        spatial_base_channels: int = 32,
    ):
        super().__init__()
        self.global_encoder = PatientEncoder(
            image_size=image_size,
            latent_dim=global_dim,
        )
        self.spatial_encoder = SpatialCXREncoder(
            latent_channels=spatial_channels,
            base_channels=spatial_base_channels,
        )
        self.cxr_decoder = ConditionalSpatialCXRDecoder(
            latent_channels=spatial_channels,
            base_channels=spatial_base_channels,
            condition_dim=global_dim,
        )
        self.ehr_decoder = TabularEHRDecoder(latent_dim=global_dim)
        self.note_decoder = NotesDecoder(
            vocab_size=vocab_size,
            latent_dim=global_dim,
            max_len=max_note_tokens,
        )
        self._train_global_backbones = True
        self._train_spatial_base = True

    def encode(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_global = self.global_encoder(batch)
        cxr_pixel = imagenet_denormalize(batch["cxr"])
        z_spatial = self.spatial_encoder(cxr_pixel)
        return z_global, z_spatial, cxr_pixel

    def reconstruct_cxr(
        self,
        batch: dict,
        condition_mode: str = "correct",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_global, z_spatial, cxr_pixel = self.encode(batch)
        if condition_mode == "zero":
            condition = torch.zeros_like(z_global)
        elif condition_mode == "shuffle":
            condition = torch.roll(z_global, shifts=1, dims=0)
        elif condition_mode == "correct":
            condition = z_global
        else:
            raise ValueError(f"unknown condition mode: {condition_mode}")
        recon = self.cxr_decoder(z_spatial, condition)
        return recon, z_global, z_spatial, cxr_pixel

    def freeze_stage1_pretrained_parts(
        self,
        train_global_backbones: bool = False,
        train_spatial_base: bool = False,
    ) -> None:
        """Stable S1 defaults: retain S0 detail and pretrained backbones."""
        self._train_global_backbones = train_global_backbones
        self._train_spatial_base = train_spatial_base
        if not train_global_backbones:
            for parameter in self.global_encoder.cxr.parameters():
                parameter.requires_grad_(False)
            for parameter in self.global_encoder.note.parameters():
                parameter.requires_grad_(False)

        if not train_spatial_base:
            for parameter in self.spatial_encoder.parameters():
                parameter.requires_grad_(False)
            self.cxr_decoder.freeze_pretrained_base()

    def set_stage1_train_mode(self) -> None:
        """Train adapters while keeping frozen backbones deterministic."""
        self.train()
        if not self._train_global_backbones:
            self.global_encoder.cxr.eval()
            self.global_encoder.note.eval()
        if not self._train_spatial_base:
            self.spatial_encoder.eval()


def load_s0_spatial_checkpoint(
    model: SpatialMultimodalAutoencoder,
    checkpoint_path: str,
) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state["model"]
    encoder_weights = {
        key.removeprefix("encoder."): value
        for key, value in weights.items()
        if key.startswith("encoder.")
    }
    decoder_weights = {
        key.removeprefix("decoder."): value
        for key, value in weights.items()
        if key.startswith("decoder.")
    }
    model.spatial_encoder.load_state_dict(encoder_weights)
    missing, unexpected = model.cxr_decoder.load_state_dict(
        decoder_weights, strict=False
    )
    if unexpected:
        raise RuntimeError(f"unexpected S0 decoder keys: {unexpected}")
    if missing and not all(key.startswith("film_") for key in missing):
        raise RuntimeError(f"unexpected missing S0 decoder keys: {missing}")
    return state


def load_multimodal_checkpoint(
    model: SpatialMultimodalAutoencoder,
    checkpoint_path: str,
) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    required = {"encoder", "ehr_dec", "note_dec"}
    missing = required.difference(state)
    if missing:
        raise KeyError(
            f"multimodal checkpoint is missing keys: {sorted(missing)}"
        )
    model.global_encoder.load_state_dict(state["encoder"])
    model.ehr_decoder.load_state_dict(state["ehr_dec"])
    model.note_decoder.load_state_dict(state["note_dec"])
    return state


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters()
            if parameter.requires_grad]


def checkpoint_exists(path: str) -> str:
    resolved = str(Path(path).expanduser())
    if not Path(resolved).is_file():
        raise FileNotFoundError(resolved)
    return resolved

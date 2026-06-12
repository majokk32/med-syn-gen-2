"""S2: frozen MedVAE spatial latents connected to multimodal fusion.

The image path is deliberately split into two responsibilities:

    CXR -> frozen MedVAE -> z_spatial -> frozen MedVAE decoder -> CXR
                              |
                              +-> trainable projector -> one CXR vector

The pooled CXR embedding is fused with EHR and report representations to
produce the shared patient vector z_global. A later conditional diffusion
model will learn p(z_spatial | z_global); this module does not flatten the
full spatial latent into the shared vector.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .decoder import NotesDecoder, TabularEHRDecoder
from .encoder import TabularEHREncoder


class MedVAELatentProjector(nn.Module):
    """Pool a MedVAE spatial map into one continuous CXR embedding."""

    def __init__(
        self,
        latent_channels: int = 3,
        output_dim: int = 768,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(self.body(latent))


class MedVAEFusionEncoder(nn.Module):
    """Fuse one CXR vector with EHR and report representations."""

    def __init__(
        self,
        latent_dim: int = 512,
        medvae_channels: int = 3,
        d_ehr: int = 192,
        ehr_layers: int = 2,
        ehr_heads: int = 4,
        note_hf_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        fusion_dim: int = 768,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
    ):
        super().__init__()
        from transformers import AutoModel

        self.cxr_projector = MedVAELatentProjector(
            latent_channels=medvae_channels,
            output_dim=fusion_dim,
        )
        self.ehr = TabularEHREncoder(
            d_model=d_ehr,
            n_layers=ehr_layers,
            n_heads=ehr_heads,
        )
        self.note = AutoModel.from_pretrained(note_hf_name)
        self.proj_ehr = nn.Linear(d_ehr, fusion_dim)
        self.proj_note = nn.Linear(
            self.note.config.hidden_size, fusion_dim
        )

        self.patient_tok = nn.Parameter(
            torch.zeros(1, 1, fusion_dim)
        )
        nn.init.trunc_normal_(self.patient_tok, std=0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=4 * fusion_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer, fusion_layers
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, latent_dim),
        )

    def forward(
        self,
        batch: dict,
        z_spatial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cxr_embedding = self.cxr_projector(z_spatial)
        ehr_tok = self.proj_ehr(self.ehr(
            batch["ehr_binary"],
            batch["ehr_cont"],
            batch["ehr_cont_mask"],
        ))
        note_out = self.note(
            input_ids=batch["note_input_ids"],
            attention_mask=batch["note_attn_mask"],
        )
        note_tok = self.proj_note(note_out.last_hidden_state)

        batch_size = cxr_embedding.size(0)
        patient = self.patient_tok.expand(batch_size, -1, -1)
        fusion_sequence = torch.cat(
            [patient, cxr_embedding.unsqueeze(1), ehr_tok, note_tok],
            dim=1,
        )
        padding_mask = torch.zeros(
            batch_size,
            fusion_sequence.size(1),
            dtype=torch.bool,
            device=fusion_sequence.device,
        )
        padding_mask[:, -note_tok.size(1):] = (
            ~batch["note_attn_mask"].bool()
        )
        hidden = self.fusion(
            fusion_sequence, src_key_padding_mask=padding_mask
        )
        z_global = self.to_latent(self.norm(hidden[:, 0]))
        return z_global, cxr_embedding


class MedVAEFusionAutoencoder(nn.Module):
    """Frozen MedVAE reconstruction plus trainable shared-latent fusion."""

    def __init__(
        self,
        vocab_size: int,
        medvae_model: str = "medvae_4_3_2d",
        max_note_tokens: int = 256,
        global_dim: int = 512,
    ):
        super().__init__()
        try:
            from medvae import MVAE
        except ImportError as exc:
            raise ImportError(
                "MedVAE is required for S2. Install it in the isolated "
                "environment with `python -m pip install medvae`."
            ) from exc

        channels = {
            "medvae_4_3_2d": 3,
            "medvae_8_4_2d": 4,
        }
        if medvae_model not in channels:
            raise ValueError(
                "S2 currently supports medvae_4_3_2d or medvae_8_4_2d"
            )
        self.medvae_name = medvae_model
        self.medvae = MVAE(
            model_name=medvae_model, modality="xray"
        )
        self.medvae.requires_grad_(False)
        self.medvae.eval()

        self.global_encoder = MedVAEFusionEncoder(
            latent_dim=global_dim,
            medvae_channels=channels[medvae_model],
        )
        self.ehr_decoder = TabularEHRDecoder(latent_dim=global_dim)
        self.note_decoder = NotesDecoder(
            vocab_size=vocab_size,
            latent_dim=global_dim,
            max_len=max_note_tokens,
        )
        # This compact target prevents the shared vector from ignoring CXR.
        self.cxr_summary_decoder = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, channels[medvae_model] * 8 * 8),
        )
        self.medvae_channels = channels[medvae_model]

    @torch.no_grad()
    def encode_spatial(self, image: torch.Tensor) -> torch.Tensor:
        normalized = image.mul(2.0).sub(1.0)
        distribution = self.medvae.model.encode(normalized)
        return distribution.mode()

    @torch.no_grad()
    def decode_spatial(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.medvae.model.decode(latent)
        return decoded.add(1.0).div(2.0).clamp(0.0, 1.0)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        z_spatial = self.encode_spatial(batch["cxr"])
        z_global, cxr_embedding = self.global_encoder(
            batch, z_spatial.detach()
        )
        reconstruction = self.decode_spatial(z_spatial)
        target_summary = torch.nn.functional.adaptive_avg_pool2d(
            z_spatial.detach(), (8, 8)
        ).flatten(1)
        predicted_summary = self.cxr_summary_decoder(z_global)
        return {
            "z_global": z_global,
            "z_spatial": z_spatial,
            "cxr_embedding": cxr_embedding,
            "cxr_reconstruction": reconstruction,
            "cxr_summary_target": target_summary,
            "cxr_summary_pred": predicted_summary,
        }

    def freeze_pretrained_backbones(self) -> None:
        self.medvae.requires_grad_(False)
        self.global_encoder.note.requires_grad_(False)
        self.global_encoder.ehr.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.medvae.eval()
        if not any(
            parameter.requires_grad
            for parameter in self.global_encoder.note.parameters()
        ):
            self.global_encoder.note.eval()
        if not any(
            parameter.requires_grad
            for parameter in self.global_encoder.ehr.parameters()
        ):
            self.global_encoder.ehr.eval()
        return self


def load_legacy_multimodal_weights(
    model: MedVAEFusionAutoencoder,
    checkpoint_path: str,
) -> dict:
    """Reuse EHR/report/fusion weights while replacing the old ViT CXR path."""
    state = torch.load(checkpoint_path, map_location="cpu")
    required = {"encoder", "ehr_dec", "note_dec"}
    missing_blocks = required.difference(state)
    if missing_blocks:
        raise KeyError(
            "multimodal checkpoint is missing blocks: "
            f"{sorted(missing_blocks)}"
        )

    encoder_state = state["encoder"]
    compatible = {
        key: value
        for key, value in encoder_state.items()
        if not (
            key.startswith("cxr.")
            or key.startswith("proj_cxr.")
        )
    }
    missing, unexpected = model.global_encoder.load_state_dict(
        compatible, strict=False
    )
    if unexpected:
        raise RuntimeError(
            f"unexpected legacy encoder keys: {unexpected}"
        )
    if not all(key.startswith("cxr_projector.") for key in missing):
        raise RuntimeError(
            f"unexpected missing fusion keys: {missing}"
        )
    model.ehr_decoder.load_state_dict(state["ehr_dec"])
    model.note_decoder.load_state_dict(state["note_dec"])
    return state


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

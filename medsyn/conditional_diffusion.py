"""Conditional diffusion from a shared patient vector to a spatial CXR latent."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings used by DDPM-style U-Nets."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualConditionBlock(nn.Module):
    """Residual convolution block modulated by time and patient condition."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(embedding)).chunk(2, dim=-1)
        x = self.norm2(x)
        x = x * (1.0 + scale[:, :, None, None])
        x = x + shift[:, :, None, None]
        x = self.conv2(F.silu(x))
        return x + residual


class SpatialAttention(nn.Module):
    """Self-attention at the U-Net bottleneck resolution."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        sequence = (
            self.norm(x).flatten(2).transpose(1, 2)
        )
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False
        )
        attended = (
            attended.transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        return x + attended


class ConditionalSpatialUNet(nn.Module):
    """Compact U-Net predicting noise in a MedVAE spatial latent."""

    def __init__(
        self,
        latent_channels: int = 3,
        condition_dim: int = 512,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        multipliers = tuple(channel_mults)
        if not multipliers:
            raise ValueError("channel_mults must not be empty")
        channels = [base_channels * value for value in multipliers]
        emb_dim = 4 * base_channels
        self.condition_dim = condition_dim
        self.condition_dropout = condition_dropout
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))
        self.input = nn.Conv2d(
            latent_channels, channels[0], 3, padding=1
        )

        down_blocks = []
        current = channels[0]
        for index, output_channels in enumerate(channels):
            down_blocks.append(nn.ModuleDict({
                "block1": ResidualConditionBlock(
                    current, output_channels, emb_dim
                ),
                "block2": ResidualConditionBlock(
                    output_channels, output_channels, emb_dim
                ),
                "downsample": (
                    nn.Conv2d(
                        output_channels, output_channels, 4,
                        stride=2, padding=1,
                    )
                    if index < len(channels) - 1
                    else nn.Identity()
                ),
            }))
            current = output_channels
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid1 = ResidualConditionBlock(current, current, emb_dim)
        self.mid_attention = SpatialAttention(current)
        self.mid2 = ResidualConditionBlock(current, current, emb_dim)

        up_blocks = []
        for index in reversed(range(len(channels))):
            output_channels = channels[index]
            up_blocks.append(nn.ModuleDict({
                "block1": ResidualConditionBlock(
                    current + output_channels, output_channels, emb_dim
                ),
                "block2": ResidualConditionBlock(
                    output_channels, output_channels, emb_dim
                ),
                "upsample": (
                    nn.ConvTranspose2d(
                        output_channels, channels[index - 1], 4,
                        stride=2, padding=1,
                    )
                    if index > 0 else nn.Identity()
                ),
            }))
            current = (
                channels[index - 1] if index > 0 else output_channels
            )
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_norm = nn.GroupNorm(
            _group_count(channels[0]), channels[0]
        )
        self.output = nn.Conv2d(
            channels[0], latent_channels, 3, padding=1
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _embedding(
        self,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.size(1) != self.condition_dim:
            raise ValueError(
                f"condition must have shape (B, {self.condition_dim})"
            )
        drop = torch.zeros(
            condition.size(0), dtype=torch.bool, device=condition.device
        )
        if force_unconditional:
            drop.fill_(True)
        elif self.training and self.condition_dropout > 0:
            drop = (
                torch.rand(condition.size(0), device=condition.device)
                < self.condition_dropout
            )
        null = self.null_condition[None].expand_as(condition)
        condition = torch.where(drop[:, None], null, condition)
        condition = self.condition_norm(condition)
        time = timestep_embedding(
            timesteps, self.time_mlp[0].in_features
        )
        return self.time_mlp(time) + self.condition_mlp(condition)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        embedding = self._embedding(
            timesteps, condition, force_unconditional
        )
        x = self.input(noisy_latent)
        skips = []
        for blocks in self.down_blocks:
            x = blocks["block1"](x, embedding)
            x = blocks["block2"](x, embedding)
            skips.append(x)
            x = blocks["downsample"](x)

        x = self.mid1(x, embedding)
        x = self.mid_attention(x)
        x = self.mid2(x, embedding)

        for blocks in self.up_blocks:
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="nearest"
                )
            x = torch.cat([x, skip], dim=1)
            x = blocks["block1"](x, embedding)
            x = blocks["block2"](x, embedding)
            x = blocks["upsample"](x)
        return self.output(F.silu(self.output_norm(x)))


def cosine_beta_schedule(
    timesteps: int,
    offset: float = 0.008,
) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5
    ).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


class ConditionalLatentDiffusion(nn.Module):
    """DDPM training and DDIM sampling around a conditional spatial U-Net."""

    def __init__(
        self,
        denoiser: ConditionalSpatialUNet,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.denoiser = denoiser
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer(
            "sqrt_alpha_bars", alpha_bars.sqrt()
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            (1.0 - alpha_bars).sqrt(),
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return values.gather(0, timesteps).view(
            timesteps.size(0), *([1] * (target.ndim - 1))
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self._extract(
                self.sqrt_alpha_bars, timesteps, clean
            ) * clean
            + self._extract(
                self.sqrt_one_minus_alpha_bars, timesteps, clean
            ) * noise
        )

    def training_loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        min_snr_gamma: float | None = 5.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if timesteps is None:
            timesteps = torch.randint(
                0, self.timesteps, (clean.size(0),), device=clean.device
            )
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.add_noise(clean, noise, timesteps)
        prediction = self.denoiser(noisy, timesteps, condition)
        per_sample = F.mse_loss(
            prediction, noise, reduction="none"
        ).flatten(1).mean(1)
        if min_snr_gamma is not None:
            alpha_bar = self.alpha_bars.gather(0, timesteps)
            snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
            weights = snr.clamp(max=min_snr_gamma) / snr.clamp_min(1e-8)
            loss = (per_sample * weights).mean()
        else:
            loss = per_sample.mean()
        return loss, {
            "noise_mse": loss.detach(),
            "prediction_std": prediction.detach().std(),
            "clean_std": clean.detach().std(),
        }

    @torch.inference_mode()
    def sample(
        self,
        condition: torch.Tensor,
        latent_shape: tuple[int, int, int],
        steps: int = 50,
        guidance_scale: float = 1.5,
        eta: float = 0.0,
        clip_denoised: float | None = 5.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not 1 <= steps <= self.timesteps:
            raise ValueError("steps must be between 1 and diffusion timesteps")
        shape = (condition.size(0), *latent_shape)
        x = (
            initial_noise.clone()
            if initial_noise is not None
            else torch.randn(shape, device=condition.device)
        )
        if tuple(x.shape) != shape:
            raise ValueError(f"initial_noise must have shape {shape}")
        schedule = torch.linspace(
            self.timesteps - 1, 0, steps,
            device=condition.device,
        ).round().long()

        was_training = self.training
        self.eval()
        for index, timestep in enumerate(schedule):
            t = torch.full(
                (condition.size(0),),
                int(timestep.item()),
                device=condition.device,
                dtype=torch.long,
            )
            conditional = self.denoiser(x, t, condition)
            if guidance_scale == 1.0:
                predicted_noise = conditional
            else:
                unconditional = self.denoiser(
                    x, t, condition, force_unconditional=True
                )
                predicted_noise = (
                    unconditional
                    + guidance_scale * (conditional - unconditional)
                )

            alpha = self.alpha_bars[timestep]
            next_timestep = (
                schedule[index + 1] if index + 1 < len(schedule) else None
            )
            next_alpha = (
                self.alpha_bars[next_timestep]
                if next_timestep is not None
                else x.new_tensor(1.0)
            )
            predicted_clean = (
                x - (1.0 - alpha).sqrt() * predicted_noise
            ) / alpha.sqrt()
            if clip_denoised is not None:
                predicted_clean = predicted_clean.clamp(
                    -clip_denoised, clip_denoised
                )
            sigma = eta * torch.sqrt(
                ((1.0 - next_alpha) / (1.0 - alpha))
                * (1.0 - alpha / next_alpha)
            ).clamp_min(0.0)
            direction = (
                1.0 - next_alpha - sigma.square()
            ).clamp_min(0.0).sqrt() * predicted_noise
            random_noise = (
                torch.randn_like(x)
                if next_timestep is not None else torch.zeros_like(x)
            )
            x = (
                next_alpha.sqrt() * predicted_clean
                + direction
                + sigma * random_noise
            )
        self.train(was_training)
        return x


class ChannelNormalizer(nn.Module):
    """Per-channel affine normalization for high-variance MedVAE latents."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be matching 1D tensors")
        self.register_buffer("mean", mean.float().view(1, -1, 1, 1))
        self.register_buffer(
            "std", std.float().clamp_min(1e-6).view(1, -1, 1, 1)
        )

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean) / self.std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std + self.mean

"""Three modality decoders from z."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import N_BIN, N_CONT

class CXRDecoder(nn.Module):
    """ConvTranspose-style image decoder: z (512) → CXR (3, 224, 224).
    
    没有 skip connections (信息瓶颈仍在 z),所以重建会有模糊感,但能看出
    胸廓 / 肺野 / 心影 这种粗结构。如果要更高保真度,改用 UNet skip 或
    latent diffusion。
    """
    
    def __init__(self, latent_dim: int = 512, base_channels: int = 256,
                 out_size: int = 224):
        super().__init__()
        # 224 = 7 * 32, so 5 doublings from 7×7
        self.seed_size = 7
        self.seed_channels = base_channels
        
        # z → 小空间 seed feature map
        self.seed = nn.Linear(
            latent_dim, base_channels * self.seed_size * self.seed_size
        )
        
        # 上采样块: Upsample + Conv + GN + SiLU (DDPM 风格)
        def up_block(c_in, c_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
            )
        
        c = base_channels
        self.up1 = up_block(c,        c)        # 7→14    256ch
        self.up2 = up_block(c,        c // 2)   # 14→28   128ch
        self.up3 = up_block(c // 2,   c // 4)   # 28→56    64ch
        self.up4 = up_block(c // 4,   c // 8)   # 56→112   32ch
        self.up5 = up_block(c // 8,   c // 16)  # 112→224  16ch
        
        # 输出层: 1×1 conv 映射到 3 通道
        self.out = nn.Conv2d(c // 16, 3, kernel_size=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.seed(z).view(
            -1, self.seed_channels, self.seed_size, self.seed_size,
        )
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.out(x)                       # (B, 3, 224, 224)
    
    def loss(self, real_cxr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """L1 重建损失。real_cxr 应该已经是 ImageNet 归一化后的 (B,3,224,224)。"""
        fake = self.forward(z)
        return F.l1_loss(fake, real_cxr)


class TabularEHRDecoder(nn.Module):
    """z → (9 binary logits, 5 continuous values, 5 missingness logits)."""
    def __init__(self, latent_dim=512, d_model=384):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, 2 * d_model), nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head_binary = nn.Linear(d_model, N_BIN)
        self.head_cont   = nn.Linear(d_model, N_CONT)
        self.head_miss   = nn.Linear(d_model, N_CONT)

    def loss(self, binary, cont, mask, z):
        h = self.body(z)
        l_bin  = F.binary_cross_entropy_with_logits(self.head_binary(h), binary)
        mf = mask.float()
        cont_diff = (self.head_cont(h) - cont) ** 2
        l_cont = (cont_diff * mf).sum() / mf.sum().clamp(min=1.0)
        l_miss = F.binary_cross_entropy_with_logits(self.head_miss(h), mf)
        return l_bin + l_cont + 0.1 * l_miss


class NotesDecoder(nn.Module):
    """Small in-house causal LM (4 layers, 384 d_model) conditioned on z.
    NOT a state-of-the-art generator; just enough to give note reconstruction
    a meaningful gradient signal. Swap for Meditron later if needed."""
    def __init__(self, vocab_size, latent_dim=512, d_model=384,
                 n_layers=4, n_heads=6, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.cond = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight                # weight-tied
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len + 1, max_len + 1, dtype=torch.bool),
                       diagonal=1),
            persistent=False,
        )

    def forward(self, input_ids, attention_mask, z):
        B, T = input_ids.shape
        pos = torch.arange(T + 1, device=z.device)[None].expand(B, -1)
        x = torch.cat([
            self.cond(z).unsqueeze(1),
            self.emb(input_ids),
        ], dim=1) + self.pos(pos)
        mask = self.causal_mask[:T + 1, :T + 1]
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=z.device),
            ~attention_mask.bool(),
        ], dim=1)
        h = self.body(x, mask=mask, src_key_padding_mask=kpm)
        return self.head(h[:, 1:])                        # (B, T, V)

    def loss(self, input_ids, attention_mask, z):
        logits = self.forward(input_ids, attention_mask, z)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        # predict t+1 from t
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
"""Multi-modal patient encoder — fuses CXR + tabular EHR + clinical note → z.

Components
----------
- CXR  : timm ViT-B/16 (ImageNet pretrained)
- EHR  : FT-Transformer (Feature Tokenizer + Transformer) — each field is
         its own token, self-attention models field interactions.
- Note : HuggingFace ClinicalBERT
- Fuse : 2-layer Transformer over the concatenation of all modality tokens,
         with a learnable [PATIENT] token at position 0. The patient token's
         hidden state at the output is projected to the latent z.

Input batch dict (produced by WideTripleDataset.__getitem__):
    cxr            : (B, 3, H, W)        float32
    ehr_binary     : (B, N_BIN)          float32 in {0, 1}
    ehr_cont       : (B, N_CONT)         float32 (z-scored, NaN → 0)
    ehr_cont_mask  : (B, N_CONT)         bool (True if value was present)
    note_input_ids : (B, T)              int64
    note_attn_mask : (B, T)              int64

Output: (B, latent_dim) — the patient latent z.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from transformers import AutoModel

from .schema import N_BIN, N_CONT


# ===========================================================================
# Tabular EHR encoder — FT-Transformer style
# ===========================================================================
class TabularEHREncoder(nn.Module):
    """Feature Tokenizer + Transformer over (binary, continuous) tabular EHR.

    Each binary field becomes 1 token via an independent Embedding.
    Each continuous field becomes 1 token via an independent Linear over
    (value, missingness_flag).
    A CLS token is prepended, all tokens get a field-identity embedding,
    then we run a small Transformer.

    Output: (B, N_BIN + N_CONT + 1, d_model)
            — token 0 is CLS, tokens 1..N_BIN are binary fields,
              tokens N_BIN+1..end are continuous fields.

    Downstream (CrossModalFusion) can use all tokens or pool to one.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = d_model

        # Per-binary-field embedding lookup (vocab size 2 = {0, 1})
        # Each field gets its OWN embedding so "gender_F=1" is encoded
        # differently from "dx_pneumonia=1".
        self.bin_emb = nn.ModuleList([
            nn.Embedding(2, d_model) for _ in range(N_BIN)
        ])

        # Per-continuous-field projection. Input is (value, mask_flag), both
        # floats; output is a d_model token.  Passing the mask flag means the
        # model knows "0.0 because NaN" is different from "0.0 because measured".
        self.cont_proj = nn.ModuleList([
            nn.Linear(2, d_model) for _ in range(N_CONT)
        ])

        # Field identity embedding (acts like BERT's segment embedding).
        # Without this, attention can't tell tokens apart beyond position.
        self.field_emb = nn.Embedding(N_BIN + N_CONT, d_model)

        # CLS token at position 0 (used by downstream if it wants a single
        # summary vector).
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        # Small Transformer (2 layers, 4 heads is plenty for 14+1 tokens).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        binary: torch.Tensor,        # (B, N_BIN)   float in {0, 1}
        cont: torch.Tensor,          # (B, N_CONT)  float (z-scored)
        mask: torch.Tensor,          # (B, N_CONT)  bool
    ) -> torch.Tensor:
        B = binary.size(0)
        device = binary.device

        # ----- 1. Tokenize each field -----
        tokens = []
        # Binary fields → Embedding(2, d) per field
        for i in range(N_BIN):
            idx = binary[:, i].long()                           # (B,) in {0,1}
            tokens.append(self.bin_emb[i](idx))                 # (B, d_model)

        # Continuous fields → Linear over (value, mask) per field
        for i in range(N_CONT):
            pair = torch.stack(
                [cont[:, i], mask[:, i].float()], dim=-1,
            )                                                   # (B, 2)
            tokens.append(self.cont_proj[i](pair))              # (B, d_model)

        x = torch.stack(tokens, dim=1)                          # (B, N, d_model)

        # ----- 2. Add field-identity embedding -----
        # field_id 0..N_BIN-1 = binary fields, N_BIN..N_BIN+N_CONT-1 = cont
        field_ids = torch.arange(N_BIN + N_CONT, device=device)
        x = x + self.field_emb(field_ids)[None]                 # broadcast over batch

        # ----- 3. Prepend CLS token -----
        cls = self.cls.expand(B, -1, -1)                        # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                          # (B, N+1, d_model)

        # ----- 4. Self-attention -----
        h = self.body(x)
        h = self.norm(h)
        return h                                                # (B, N+1, d_model)


# ===========================================================================
# Top-level Patient Encoder
# ===========================================================================
class PatientEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        latent_dim: int = 512,
        d_ehr: int = 192,
        ehr_layers: int = 2,
        ehr_heads: int = 4,
        cxr_pretrained: bool = True,
        note_hf_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        fusion_layers: int = 2,
        fusion_heads: int = 8,
    ):
        super().__init__()

        # ---- 1. CXR backbone (timm ViT-B/16) ----
        self.cxr = timm.create_model(
            "vit_base_patch16_224",
            pretrained=cxr_pretrained,
            num_classes=0,
            img_size=image_size,
        )
        d_cxr = self.cxr.num_features                           # 768

        # ---- 2. Tabular EHR encoder (FT-Transformer) ----
        # Outputs (B, N_BIN + N_CONT + 1, d_ehr) — 15 tokens total.
        self.ehr = TabularEHREncoder(
            d_model=d_ehr,
            n_layers=ehr_layers,
            n_heads=ehr_heads,
        )

        # ---- 3. Note encoder (ClinicalBERT) ----
        self.note = AutoModel.from_pretrained(note_hf_name)
        d_note = self.note.config.hidden_size                   # 768

        # ---- 4. Cross-modal fusion ----
        d_fuse = max(d_cxr, d_ehr, d_note)
        self.proj_cxr = nn.Linear(d_cxr, d_fuse)
        self.proj_ehr = nn.Linear(d_ehr, d_fuse)
        self.proj_note = nn.Linear(d_note, d_fuse)

        # Learnable [PATIENT] token; its final hidden state IS the latent.
        self.patient_tok = nn.Parameter(torch.zeros(1, 1, d_fuse))
        nn.init.trunc_normal_(self.patient_tok, std=0.02)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_fuse,
            nhead=fusion_heads,
            dim_feedforward=4 * d_fuse,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, fusion_layers)
        self.norm = nn.LayerNorm(d_fuse)

        # Final projection to the latent.
        self.to_latent = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, latent_dim),
        )

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """batch dict → z ∈ ℝ^(B × latent_dim)."""

        # ---- CXR → tokens (B, 197, d_fuse) ----
        cxr_h = self.cxr.forward_features(batch["cxr"])
        if cxr_h.dim() == 2:
            cxr_h = cxr_h.unsqueeze(1)
        cxr_tok = self.proj_cxr(cxr_h)

        # ---- EHR → tokens (B, 15, d_fuse) ----
        # NOTE: FT-Transformer outputs 15 tokens (1 CLS + 9 binary + 5 cont).
        # We project all of them and feed them into the fusion stage so the
        # fusion attention can use field-level information.
        ehr_h = self.ehr(
            batch["ehr_binary"],
            batch["ehr_cont"],
            batch["ehr_cont_mask"],
        )                                                       # (B, 15, d_ehr)
        ehr_tok = self.proj_ehr(ehr_h)                          # (B, 15, d_fuse)

        # ---- Note → tokens (B, T, d_fuse) ----
        note_out = self.note(
            input_ids=batch["note_input_ids"],
            attention_mask=batch["note_attn_mask"],
        )
        note_tok = self.proj_note(note_out.last_hidden_state)

        # ---- Concat with [PATIENT] token at position 0 ----
        B = cxr_tok.size(0)
        pt = self.patient_tok.expand(B, -1, -1)                 # (B, 1, d_fuse)
        x = torch.cat([pt, cxr_tok, ehr_tok, note_tok], dim=1)

        # ---- Build key padding mask (only notes can have padding) ----
        # Position layout in `x`:
        #   0                                : [PATIENT]
        #   1 .. cxr_len                     : CXR tokens (always full)
        #   cxr_len+1 .. cxr_len+ehr_len     : EHR tokens (always full)
        #   cxr_len+ehr_len+1 .. end         : Note tokens (may be padded)
        pad = torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)
        n_note = note_tok.size(1)
        pad[:, -n_note:] = ~batch["note_attn_mask"].bool()

        # ---- Fuse ----
        h = self.fusion(x, src_key_padding_mask=pad)
        h = self.norm(h[:, 0])                                  # take [PATIENT]
        return self.to_latent(h)                                # (B, latent_dim)
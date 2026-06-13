"""Inspect S2 MedVAE fusion and optionally compare saved legacy outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoTokenizer

from medsyn.data import clean_text
from medsyn.medvae_fusion import (
    MedVAEFusionAutoencoder,
    load_s2_checkpoint,
)
from medsyn.schema import (
    BINARY_COLS,
    CONTINUOUS_COLS,
    CONT_MEAN,
    CONT_STD,
)
from medsyn.spatial_cxr import ResizeAndPad, reconstruction_metrics
from medsyn.spatial_multimodal import SubjectSplitTripleDataset


class MedVAEInspectionDataset(SubjectSplitTripleDataset):
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
    width = max(110, 8 * len(label) + 12)
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def load_legacy_samples(path: Path) -> list[int]:
    report = (path / "inspect.md").read_text(encoding="utf-8")
    match = re.search(r"^samples:\s*(\[.*\])", report, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"cannot find samples list in {path / 'inspect.md'}")
    return [int(value) for value in ast.literal_eval(match.group(1))]


def legacy_reconstruction(path: Path, sample_number: int, size: int) -> np.ndarray:
    panel = Image.open(path / f"{sample_number:02d}_cxr.png").convert("RGB")
    left = panel.width // 2
    reconstruction = panel.crop((left, 0, panel.width, panel.height))
    reconstruction = reconstruction.resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(reconstruction)


def generate_note(model, tokenizer, z_global, max_tokens: int) -> str:
    bos = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
    stop = tokenizer.sep_token_id or tokenizer.eos_token_id
    generated = [bos]
    for _ in range(max_tokens):
        input_ids = torch.tensor([generated], device=z_global.device)
        attention_mask = torch.ones_like(input_ids)
        logits = model.note_decoder(input_ids, attention_mask, z_global)
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        if stop is not None and next_token == stop:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--out")
    parser.add_argument("--legacy-inspection")
    parser.add_argument(
        "--source-indices",
        default="89876,50966,76013,10364,50285",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    state = torch.load(args.ckpt, map_location="cpu")
    config = state.get("args", {})
    medvae_model = state.get(
        "medvae_model", config.get("medvae_model", "medvae_4_3_2d")
    )
    max_note_tokens = int(config.get("max_note_tokens", 256))
    global_dim = int(config.get("global_dim", 512))

    dataset = MedVAEInspectionDataset(
        args.features,
        args.cxr_root,
        tokenizer,
        image_size=args.image_size,
        max_note_tokens=max_note_tokens,
        split=None,
        train=False,
    )
    source_indices = [
        int(value.strip())
        for value in args.source_indices.split(",")
        if value.strip()
    ]
    local_indices = [
        dataset.local_index_for_source(value)
        for value in source_indices
    ]

    legacy_path = (
        Path(args.legacy_inspection)
        if args.legacy_inspection else None
    )
    if legacy_path is not None:
        legacy_samples = load_legacy_samples(legacy_path)
        if legacy_samples != source_indices:
            raise ValueError(
                "legacy samples do not match requested source indices: "
                f"{legacy_samples} != {source_indices}"
            )

    model = MedVAEFusionAutoencoder(
        vocab_size=tokenizer.vocab_size,
        medvae_model=medvae_model,
        max_note_tokens=max_note_tokens,
        global_dim=global_dim,
    )
    load_s2_checkpoint(model, args.ckpt)
    model.to(device).eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(
        args.out or f"runs/medvae_fusion_s2_inspection_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    report = [
        "# S2 MedVAE Fusion Inspection",
        "",
        f"checkpoint: `{args.ckpt}`",
        f"checkpoint step: `{state.get('step')}`",
        f"source indices: `{source_indices}`",
        "",
    ]

    for sample_number, local_index in enumerate(local_indices):
        sample = dataset[local_index]
        row = dataset.df.iloc[local_index]
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
        }
        with torch.inference_mode():
            model_output = model(batch)
            z_global = model_output["z_global"]
            reconstruction = model_output["cxr_reconstruction"]
            image_metrics = reconstruction_metrics(
                reconstruction, batch["cxr"]
            )
            ehr_hidden = model.ehr_decoder.body(z_global)
            binary_probability = torch.sigmoid(
                model.ehr_decoder.head_binary(ehr_hidden)
            )[0].cpu().numpy()
            continuous_z = model.ehr_decoder.head_cont(
                ehr_hidden
            )[0].cpu().numpy()
            observed_probability = torch.sigmoid(
                model.ehr_decoder.head_miss(ehr_hidden)
            )[0].cpu().numpy()
            generated_note = generate_note(
                model, tokenizer, z_global, args.gen_tokens
            )

        real_image = to_uint8(batch["cxr"][0])
        s2_image = to_uint8(reconstruction[0])
        columns = [add_label(real_image, "real")]
        if legacy_path is not None:
            old_image = legacy_reconstruction(
                legacy_path, sample_number, args.image_size
            )
            columns.append(add_label(old_image, "old shared-z CNN"))
        columns.extend([
            add_label(s2_image, "S2 MedVAE"),
            add_label(
                to_uint8(
                    (reconstruction[0] - batch["cxr"][0]).abs().mul(3)
                ),
                "S2 error x3",
            ),
        ])
        Image.fromarray(np.concatenate(columns, axis=1)).save(
            output / f"{sample_number:02d}_cxr_comparison.png"
        )

        real_note = (
            clean_text(row.get("impression", ""))
            or clean_text(row.get("findings", ""))
            or "[NO REPORT]"
        )
        (output / f"{sample_number:02d}_note_real.txt").write_text(
            real_note, encoding="utf-8"
        )
        (output / f"{sample_number:02d}_note_s2.txt").write_text(
            generated_note, encoding="utf-8"
        )
        old_note = None
        if legacy_path is not None:
            old_note_file = legacy_path / f"{sample_number:02d}_note_gen.txt"
            old_note = old_note_file.read_text(encoding="utf-8")
            (output / f"{sample_number:02d}_note_old.txt").write_text(
                old_note, encoding="utf-8"
            )

        report.extend([
            f"## Sample {sample_number}",
            "",
            f"`source_index={source_indices[sample_number]}`, "
            f"`subject_id={int(row['subject_id'])}`, "
            f"`study_id={int(row['study_id'])}`",
            "",
            f"![CXR comparison]({sample_number:02d}_cxr_comparison.png)",
            "",
            "### EHR",
            "",
            "| Field | Real | S2 prediction | P(observed) |",
            "|---|---:|---:|---:|",
        ])
        for field_index, field in enumerate(BINARY_COLS):
            real_value = int(row.get(field, 0) or 0)
            report.append(
                f"| {field} | {real_value} | "
                f"{binary_probability[field_index]:.3f} | - |"
            )
        for field_index, field in enumerate(CONTINUOUS_COLS):
            raw_value = row.get(field, np.nan)
            real_value = (
                f"{float(raw_value):.2f}"
                if pd.notna(raw_value) else "NaN"
            )
            prediction = (
                continuous_z[field_index] * CONT_STD[field_index]
                + CONT_MEAN[field_index]
            )
            report.append(
                f"| {field} | {real_value} | {prediction:.2f} | "
                f"{observed_probability[field_index]:.3f} |"
            )
        report.extend([
            "",
            "### Report",
            "",
            "**Real**",
            "",
            real_note,
            "",
        ])
        if old_note is not None:
            report.extend([
                "**Old shared-z model**",
                "",
                old_note,
                "",
            ])
        report.extend([
            "**S2 MedVAE fusion**",
            "",
            generated_note,
            "",
        ])

        metric_rows.append({
            "sample": sample_number,
            "source_index": source_indices[sample_number],
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "mae": float(image_metrics["mae"]),
            "psnr": float(image_metrics["psnr"]),
            "ssim": float(image_metrics["ssim"]),
            "z_global_std": float(z_global.std()),
            "z_spatial_std": float(model_output["z_spatial"].std()),
        })

    with (output / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=metric_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {
        "checkpoint": args.ckpt,
        "checkpoint_step": state.get("step"),
        "medvae_model": medvae_model,
        "source_indices": source_indices,
        "legacy_inspection": (
            str(legacy_path) if legacy_path is not None else None
        ),
        "mean_mae": float(np.mean([row["mae"] for row in metric_rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in metric_rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in metric_rows])),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "inspect.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved S2 comparison to {output}", flush=True)


if __name__ == "__main__":
    main()

# Spatial CXR Latent Experiment (S0)

这个实验独立验证：

> 把 CXR 从单个 `z ∈ R^512` 改成 `z_spatial ∈ R^(4×28×28)`，是否能改善图像重建。

当前阶段只训练：

```text
CXR 3×224×224
    ↓ spatial encoder
z_spatial 4×28×28
    ↓ spatial decoder
reconstructed CXR 3×224×224
```

它不会修改现有三模态模型，也暂时不包含 fusion、EHR、report、diffusion
或 prior。encoder 和 decoder 之间没有 skip connection，因此图像信息必须
经过 spatial latent。

## 文件

- `medsyn/spatial_cxr.py`: 数据集、空间 encoder/decoder、loss 和 metrics
- `train_spatial_cxr.py`: subject-level train/val split、训练和 best checkpoint
- `inspect_spatial_cxr.py`: test 重建图、MAE、PSNR、SSIM 和 latent 统计

## 先跑 20-step smoke test

```bash
python train_spatial_cxr.py \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --output runs/spatial_cxr_smoke \
  --batch-size 8 \
  --num-workers 4 \
  --max-steps 20 \
  --val-every 20 \
  --max-val-batches 5 \
  --limit-train-rows 1000 \
  --limit-val-rows 200
```

## 正式跑 2000 steps

```bash
python train_spatial_cxr.py \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --output runs/spatial_cxr_4x28x28 \
  --batch-size 16 \
  --num-workers 8 \
  --max-steps 2000 \
  --val-every 200
```

## 查看 test reconstruction

```bash
python inspect_spatial_cxr.py \
  --ckpt runs/spatial_cxr_4x28x28/ckpt_best.pt \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --split test \
  --n 10
```

若要和旧模型严格使用同一批 parquet 行，可以使用：

```bash
python inspect_spatial_cxr.py \
  --ckpt runs/spatial_cxr_4x28x28/ckpt_best.pt \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --split all \
  --source-indices 89876,50966,76013,10364,50285
```

这组 `--split all` 只用于和旧重建图做公平视觉对照。正式结果仍应报告
subject-level held-out test 指标。

每张输出图从左到右是：

```text
real | reconstruction | absolute error × 3
```

## 判定标准

优先看 held-out test：

- reconstruction 不再只是同一张平均胸片
- 肺野边界、心影、膈肌和局部高密度区域更清楚
- SSIM / PSNR 高于旧的 global-vector CXR decoder
- `latent_std` 不接近 0，避免 latent collapse

如果 S0 成立，下一步才做 S1：

```text
z_patient_global = fusion(CXR global, EHR, report)
z_cxr_spatial = spatial CXR encoder(CXR)

CXR decoder input:
    z_cxr_spatial + FiLM(z_patient_global)
```

之后再把 CXR spatial decoder 替换为 conditional latent diffusion。

## S1: 接回三模态 shared latent

S1 使用：

```text
normalized CXR ─┐
EHR ────────────┼→ PatientEncoder → z_patient_global
report ─────────┘

pixel CXR → spatial encoder → z_cxr_spatial

CXR decoder(z_cxr_spatial, FiLM(z_patient_global))
EHR decoder(z_patient_global)
report decoder(z_patient_global)
```

`train_spatial_multimodal.py` 会同时加载：

- S0 的 `ckpt_best.pt`
- 原三模态模型的 `ckpt_best.pt`

默认冻结 ViT、ClinicalBERT、spatial encoder 和 spatial decoder 主体，只训练
fusion/EHR encoder/EHR decoder/report decoder/FiLM。这样先验证两条 latent
路径可以稳定接在一起，不立即破坏 S0 的 CXR 重建。

验证日志还会报告：

- `shuffle_delta`: 把其他患者的 global condition 换进来后 MAE 增量
- `zero_delta`: 去掉 global condition 后 MAE 增量

正值表示正确的 `z_patient_global` 确实对 CXR reconstruction 有帮助。

S1 分开保存：

- `ckpt_best_joint.pt`: 加权三模态 validation loss 最优
- `ckpt_best_cxr.pt`: CXR validation loss 最优
- `ckpt_last.pt`: 最后一步

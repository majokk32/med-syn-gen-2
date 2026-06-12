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

## S0.5: 减少重建模糊

如果增加 latent channels 没有明显改善，优先微调 decoder 和高频 loss：

```text
bilinear upsample → nearest upsample + learned convolution
L1/edge/SSIM      → L1/edge/SSIM + multi-scale Laplacian
```

这一步不使用 skip connection，也不引入 GAN，因此不会绕过 spatial latent，
同时比 adversarial sharpening 更不容易制造不存在的医学细节。

从 4-channel S0 best checkpoint 继续微调：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="runs/spatial_cxr_sharp_4x28x28_${STAMP}"

python train_spatial_cxr.py \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --resume runs/spatial_cxr_4x28x28_20260611/ckpt_best.pt \
  --reset-best \
  --output "$OUT" \
  --latent-channels 4 \
  --upsample-mode nearest \
  --laplacian-weight 0.20 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 2600 \
  --val-every 100 \
  --seed 42 \
  2>&1 | tee "${OUT}.log"
```

旧 checkpoint 是 step 1800，因此 `max-steps 2600` 表示继续微调 800 steps。

如果 S0 成立，下一步才做 S1：

```text
z_patient_global = fusion(CXR global, EHR, report)
z_cxr_spatial = spatial CXR encoder(CXR)

CXR decoder input:
    z_cxr_spatial + FiLM(z_patient_global)
```

之后再把 CXR spatial decoder 替换为 conditional latent diffusion。

## S0.6: 预训练医学 VAE 对比

在继续训练 diffusion 前，先检查图像 autoencoder 本身的重建上限。
这里使用 Stanford MIMI 的 MedVAE，并保持 continuous spatial latent，
不采用离散 tokenizer。

推荐同时比较：

- `medvae_8_4_2d`: `224x224 -> 4x28x28`，与当前 spatial latent
  形状完全相同，可以作为直接替换候选。
- `medvae_4_3_2d`: `224x224 -> 3x56x56`，用来测试更低压缩率能够带来
  多大的肉眼改善。

MedVAE 依赖较多，建议在单独环境中安装，避免改变原训练环境的
PyTorch 版本：

```bash
module purge
module load python/3.11.9
python -m venv ~/envs/medvae_eval_env
source ~/envs/medvae_eval_env/bin/activate
python -m pip install --upgrade pip
python -m pip install medvae pyarrow
```

在 GPU 节点运行固定 test10 对比：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="runs/pretrained_medvae_same10_${STAMP}"

python inspect_pretrained_medvae_cxr.py \
  --features ../datasets/vlm_radiology_report_generation/output/mimic_cxr_features.parquet \
  --cxr-root ../datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org \
  --local-ckpt runs/spatial_cxr_sharp_4x28x28_20260612_094858/ckpt_best.pt \
  --medvae-models medvae_8_4_2d,medvae_4_3_2d \
  --split test \
  --source-indices 10615,89249,11071,75129,50337,49949,80335,11806,22798,99360 \
  --out "$OUT" \
  2>&1 | tee "${OUT}.log"
```

第一次运行会从官方 Hugging Face repository 下载 MedVAE 权重。脚本使用
posterior mode 而非随机采样，保证不同模型在同一批样本上的重建比较可重复。
输出包含逐样本 comparison panel、`metrics.csv` 和 `summary.json`。

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

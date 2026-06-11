# Med-Syn-Gen — 多模态合成病人生成（最小可运行版本）

基于 MIMIC-IV / MIMIC-CXR / MIMIC-IV-Note 的多模态病人表示学习与重建。
当前是 **stage 1（编码 + 重建）** 的最小骨架，验证三模态融合是否可行。

---

## 整体架构

```
Parquet 一行（21 列）
        │
        ▼
WideTripleDataset
        │
   ┌────┼────┐
   ▼    ▼    ▼
 ViT  MLP  ClinicalBERT
(CXR) (EHR) (Notes)
   │    │    │
   └────┼────┘
        ▼
CrossModalFusion ([PATIENT] + 2 层 Transformer)
        │
        ▼
   z ∈ ℝ⁵¹²        ← 病人表示瓶颈
        │
   ┌────┼────┐
   ▼    ▼    ▼
 CXR  EHR  Notes
 Dec  Dec  Dec
   │    │    │
   ▼    ▼    ▼
 64×64 9bin+5cnt token logits
   │    │    │
   ▼    ▼    ▼
 MSE  BCE+MSE  CE
   └────┬─────┘
        ▼
   loss = l_cxr + l_ehr + l_note
```

---

## 已完成部分

### 1. 数据预处理 (`Data.ipynb`)

输入：MIMIC-IV 3.1 + MIMIC-CXR-JPG 2.1.0 原始 CSV / JPG。

输出：`mimic_cxr_features.parquet`，**116,129 行 × 21 列**。每行 = 一张 CXR + 周围 ±24h 的临床快照。

包含字段：

| 类别 | 字段 | 数量 |
|---|---|---|
| 标识 | subject_id, study_id, dicom_id, hadm_id, study_datetime | 5 |
| Demographics | anchor_age, gender_F | 2 |
| 疾病标签（ICD） | dx_pneumonia, dx_pneumothorax, dx_chf, dx_pleural_effusion, dx_atelectasis | 5 |
| 医疗支持 | is_intubated, has_pacemaker, on_ventilator | 3 |
| 生命体征 | spo2, resp_rate | 2 |
| 实验室 | wbc, bnp | 2 |
| 放射学报告 | findings, impression | 2 |

**关键设计**：
- CXR ↔ admission 通过时间窗（`admittime ≤ study_datetime ≤ dischtime`）匹配
- 实验室/生命体征采"距 CXR 时刻最近的一次值"（`closest_value_in_window`，±24h 窗口）
- 放射学报告用正则解析 FINDINGS / IMPRESSION 两段

**数据缩水**：377K 张原始 dicom → 116K（70% 缩水）。主要丢在"门诊 CXR 无对应住院"。

### 2. 最小训练管线 (`train_minimal.py`)

单文件 413 行，无 synthmed 包依赖。

**模型组成**（总 ~212M 参数）：

| 模块 | 实现 | 参数 |
|---|---|---|
| CXR encoder | timm ViT-B/16（pretrained） | 86M |
| EHR encoder | 19→768→384 MLP | <1M |
| Notes encoder | emilyalsentzer/Bio_ClinicalBERT | 110M |
| Cross-modal fusion | [PATIENT] token + 2 层 self-attn → 512 dim | ~4M |
| CXR decoder | MLP → 64×64 重建 | 8M |
| Tabular EHR decoder | MLP → 9 binary + 5 cont + 5 missingness | 1M |
| Notes decoder | 4 层 causal Transformer + weight-tied LM head | 3M |

**损失**：`loss = MSE(cxr) + (BCE+MSE+0.1·BCE)(ehr) + CE(notes)`

**硬件需求**：
- GPU 显存：5-12GB（batch 8-32）
- CPU RAM：< 5GB
- 一张 A100 / V100 / 甚至 RTX 3090 都够

### 3. 配套工具

| 脚本 | 作用 |
|---|---|
| `check_missing_cxr.py` | 检查 parquet 里有多少行对应的 jpg 不在磁盘上 |
| `inspect_minimal.py` | 加载 ckpt 在 N 个真实病人上跑闭环，输出 CXR 重建对比图 + 表格预测 + 生成报告 |

---

## 当前效果

**已验证**：
- ✅ 三模态数据流通畅，batch 形状正确
- ✅ 训练 loss 能稳定下降，无 NaN / Inf
- ✅ 编码器 forward + 三个 decoder loss + 反向传播无错误
- ✅ ckpt 保存与加载流程跑通

**未充分验证**：
- ⚠️ 长训（>5000 步）下的 val loss 趋势
- ⚠️ z 的多样性（是否塌缩为单点）
- ⚠️ 三模态重建质量（CXR 重建是否高于平均图基线、Notes 生成是否成英文）

**已知局限**：
- CXR decoder 只是 MLP → 64×64，不是真正的生成模型，重建质量天花板低
- Notes decoder 从零训练，4 层 Transformer 在 200-500 步内生成几乎都是 gibberish
- 没有 validation 监控，过拟合无法察觉
- 训练结束才存 ckpt，中途挂掉损失全部进度

---

## 已放弃的实现

在精简到 `train_minimal.py` 之前，有一版完整的 `synthmed/` 包含以下模块，**因依赖复杂、跑不通而放弃**：

| 模块 | 放弃原因 |
|---|---|
| RoentGen v2 latent diffusion CXR decoder | 6GB 模型下载 + diffusers 版本兼容 |
| Meditron-7B + 4-bit LoRA Notes decoder | bitsandbytes / accelerate 版本踩坑 + 14GB 显存 |
| BiomedCLIP 跨模态一致性损失 | 增加 800MB 依赖 + 复杂度 |
| Opacus DP-SGD（隐私保护） | 依赖与 PyTorch 版本耦合 |
| 因果 DAG 分区的 latent | 设计复杂、当前无法验证收益 |
| Latent diffusion prior（DiT 风格） | stage 1 不需要 |
| 三阶段训练（encoder → prior → joint） | 单阶段足够验证管线 |
| HALO 风格事件序列 EHR | 现有宽表足够 |

这些放在 `synthmed/` 目录下，**作为未来扩展参考**，不参与当前训练。

---

## 怎么跑

### 1. 数据预处理（一次性）

```bash
jupyter notebook datasets/vlm_radiology_report_generation/Data.ipynb
# 跑完得到 output/mimic_cxr_features.parquet
```

### 2. 检查数据完整性（可选但推荐）

```bash
python check_missing_cxr.py \
    --features output/mimic_cxr_features.parquet \
    --cxr-root /path/to/mimic-cxr-jpg-2.1.0.physionet.org
```

### 3. 训练

```bash
pip install torch torchvision timm pandas pillow transformers pyarrow

python train_minimal.py \
    --features output/mimic_cxr_features.parquet \
    --cxr-root /path/to/mimic-cxr-jpg-2.1.0.physionet.org \
    --batch-size 8 \
    --max-steps 1000 \
    --output runs/minimal
```

### 4. 推理 + 可视化

```bash
python inspect_minimal.py \
    --ckpt runs/minimal/ckpt.pt \
    --features output/mimic_cxr_features.parquet \
    --cxr-root /path/to/mimic-cxr-jpg \
    --n 5 \
    --out runs/minimal/inspect
```

输出 `inspect.md` + 若干 `cxr.png` + 生成报告文本。

---

## 路线图（按优先级）

### 🟡 Phase 1：训练管线稳定性（推荐立刻做）

| # | 项目 | 工作量 | 状态 |
|---|---|---|---|
| 1 | 添加 train/val/test split（按 subject_id 哈希）| 30 分钟 | ⏳ |
| 2 | 训练循环里加 validation loss 评估 | 1 小时 | ⏳ |
| 3 | 定期 checkpoint + 保存 best val ckpt | 30 分钟 | ⏳ |
| 4 | 用真实训练集统计替换 z-score 占位值 | 30 分钟 | ⏳ |

### 🟠 Phase 2：模型质量提升

| # | 项目 | 工作量 | 备注 |
|---|---|---|---|
| 6 | **InfoNCE 对比损失** 防 z 塌缩 | 1 小时 | ⭐ 强烈推荐 |
| 7 | 冻结 ViT 前 8 层 + ClinicalBERT 前 8 层 | 1 小时 | 省 30% 显存 |
| 8 | Modality dropout（10% 随机丢一种模态）| 1 小时 | 鲁棒性 |
| 9 | bf16 autocast | 30 分钟 | 1.5× 速度 |
| 10 | 学习率 warmup + cosine 调度 | 1 小时 | 训练稳定性 |
| 11 | **EHR encoder 换 FT-Transformer**（每字段一个 token + self-attn）| 半天 | 字段交互建模 |

### 🔵 Phase 3：从"重建"到"生成"（关键里程碑）

**当前管线只能重建已有病人，不能造新病人**。要做合成数据，必须加先验。

| # | 项目 | 工作量 | 备注 |
|---|---|---|---|
| 12 | **加 VAE prior**：encoder 输出 (μ, logσ²)，KL 拉向 N(0,I)，可从 N(0,I) 采样 z_new | 1 天 | ⭐⭐ 最小可行先验 |
| 13 | **加 Latent Diffusion prior**：在 z 上训 DiT 扩散，采样质量更高 | 3-5 天 | 论文级 |
| 14 | 实现 `sample.py`：从 prior 采样 → 三个 decoder 解码 → 输出合成病人 | 1 天 | 验收先验质量 |

### 🟣 Phase 4：解码器升级（让合成内容真实可用）

| # | 项目 | 工作量 | 备注 |
|---|---|---|---|
| 15 | CXR decoder 换成小 UNet（32 层）| 1 天 | 64×64 → 256×256 |
| 16 | CXR decoder 换成 RoentGen 微调 | 1 周 | SOTA CXR 质量 |
| 17 | Notes decoder 换成 GPT-2 small（prefix-tuning）| 1 天 | 流畅医学英文 |
| 18 | Notes decoder 换成 Meditron-7B + LoRA | 3-5 天 | 论文级临床文本 |

### 🟢 Phase 5：研究级特性

| # | 项目 | 工作量 | 备注 |
|---|---|---|---|
| 19 | 跨模态一致性损失（BiomedCLIP）| 2 天 | 防止三模态各说各话 |
| 20 | 因果 DAG 分区的 latent + DAG-masked attention | 1 周 | 可控反事实生成 |
| 21 | DP-SGD（Opacus）训练 prior | 3 天 | 形式化隐私保证 (ε, δ) |
| 22 | 评估指标：TSTR + MIA + 分布距离 | 3 天 | 合成数据质量论文标配 |

### 🔧 Phase 6：工程化

| # | 项目 | 工作量 | 备注 |
|---|---|---|---|
| 23 | DDP 多卡训练 | 1 天 | 加速 |
| 24 | 把 `train_minimal.py` 拆成 `medsyn/` 包 | 1 天 | 可维护性 |
| 25 | 日志切换到 wandb / tensorboard | 半天 | 实验追踪 |

---

## 关键设计决策记录

### 为什么放弃 HALO 风格事件序列

HALO 把 EHR 视作 `(token, time, value)` 三元组序列，能建模"事件发生的时间动态"。但**我们当前数据是 ±24h 窗口的聚合快照**（每个字段只保留一个值），**delta_time 全是 0**，时间维度退化。

→ 当前用 **tabular wide-table** 形式更匹配数据本身。等做完了 stage 1，如果要做纵向（多次入院）合成，再回头取真实事件流。

### 为什么 z 是 512 维而不是更小

更小的 z 会让重建任务太难，loss 高；更大的 z 容易过拟合并丧失"瓶颈"压缩效应。512 是论文中常见的中等容量选择，留有调整空间。

### 为什么三个损失直接相加

简单。生产实践里会按 loss 量级用 [Multi-Task Learning](https://arxiv.org/abs/1705.07115) 自适应加权，但 stage 1 不需要。

### 为什么不一开始就用 VAE

VAE 训练敏感（KL 权重调不好就塌缩），需要先确认 encoder 能学到有意义的 z。等 stage 1 验证 encoder 后再升级。

---

## 文件清单

```
.
├── README.md                          ← 本文件
├── Data.ipynb                         ← MIMIC 数据预处理
├── train_minimal.py                   ← 主训练脚本（单文件）
├── inspect_minimal.py                 ← 推理可视化
├── check_missing_cxr.py               ← 数据完整性检查
├── output/
│   └── mimic_cxr_features.parquet     ← 预处理产物（116K 行 × 21 列）
└── runs/
    └── minimal/
        ├── ckpt.pt                    ← 模型权重
        └── inspect/                   ← inspect_minimal.py 输出
```

---

## 依赖

```
torch>=2.0
torchvision>=0.15
timm>=0.9                # ViT backbone
transformers>=4.30       # ClinicalBERT
pandas>=2.0
pillow>=10.0
pyarrow>=12.0            # parquet
tqdm>=4.65               # check_missing_cxr 进度条
```

无需 diffusers / peft / bitsandbytes / opacus / open_clip。

---
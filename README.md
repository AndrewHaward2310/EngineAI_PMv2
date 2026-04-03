# 🤖 EngineAI PM_v2 — Humanoid Robot RL Training

> **Bài test tích hợp robot EngineAI PM_v2 (24 DOF) vào framework mjlab (MuJoCo + Warp GPU)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Warp%20GPU-green)](https://mujoco.org)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-orange)](https://arxiv.org/abs/1707.06347)
[![License](https://img.shields.io/badge/License-Private-red)]()

---

## 📊 Kết Quả Training

| Task | Iterations | Envs | GPU | Mean Reward | Thời gian |
|:-----|:----------:|:----:|:---:|:-----------:|:---------:|
| **Velocity** (đi bộ) | 20,000 | 4,096 | RTX 3090 | **65.98** | ~7h |
| **Boxing** (đấm bốc) | 30,000 | 4,096 | RTX 3090 | **~35** | ~14h |
| **Kicking** (đá) | 20,000 | 4,096 | RTX 3090 | **~32** | ~7h |
| **Dance** (nhảy) | 5,000 | 16,384 | A100 80GB | **38.27** | ~4.3h |

---

## 🎥 Demo Videos

### Velocity — Locomotion (đi bộ theo lệnh tốc độ)

https://github.com/AndrewHaward2310/EngineAI_PMv2/releases/download/v1.0/velocity_1080p.mp4

### Boxing — Motion Mimicry (bắt chước đấm bốc)

https://github.com/AndrewHaward2310/EngineAI_PMv2/releases/download/v1.0/boxing_1080p_60s.mp4

### Kicking — Motion Mimicry (bắt chước đá)

https://github.com/AndrewHaward2310/EngineAI_PMv2/releases/download/v1.0/kicking_1080p_60s.mp4

### Dance — Motion Mimicry (bắt chước nhảy, trained on A100 80GB)

https://github.com/AndrewHaward2310/EngineAI_PMv2/releases/download/v1.0/dance_1080p_60s.mp4

---

## 🏗️ Kiến Trúc Kỹ Thuật

### Robot EngineAI PM_v2

```
24 DOF Humanoid Robot
├── Legs (12 DOF)     — 6 per leg: hip yaw/roll/pitch, knee, ankle roll/pitch
├── Arms (10 DOF)     — 5 per arm: shoulder pitch/roll/yaw, elbow pitch/yaw
├── Waist (1 DOF)     — yaw rotation
└── Head (1 DOF)      — yaw rotation
```

### RL Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                   ManagerBasedRlEnv                       │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ Action   │  │ Obs      │  │ Reward                 │ │
│  │ Manager  │  │ Manager  │  │ Manager (14 terms)     │ │
│  └────┬─────┘  └────┬─────┘  └────┬───────────────────┘ │
│       │             │             │                      │
│  ┌────▼─────────────▼─────────────▼────────────────────┐ │
│  │              Scene (MuJoCo Warp on GPU)              │ │
│  │        N robots × physics step on CUDA               │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │Terminal  │  │ Event    │  │ Curriculum             │ │
│  │ Manager  │  │ Manager  │  │ Manager                │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                         ▲
                         │ RslRlVecEnvWrapper
                         ▼
                ┌──────────────────┐
                │  PPO (RSL-RL)    │
                │  Actor [512,256, │
                │  128] + Critic   │
                └──────────────────┘
```

**Key specs:**
- **Physics**: MuJoCo Warp (GPU), timestep=0.005s, decimation=4 → 50Hz control
- **Algorithm**: PPO with adaptive LR, GAE (γ=0.99, λ=0.95), clip=0.2
- **Network**: MLP Actor (135→512→256→128→24) + Critic (261→512→256→128→1), ELU activation

---

## 📁 Cấu Trúc Thư Mục

```
EngineAI_PMv2/
├── 📄 README.md                       ← Bạn đang đọc file này
│
├── 📂 docs/                           ← Tài liệu phân tích (11 files, ~3,730 dòng)
│   ├── de_bai_phan_tich.md            # Đề bài + phân tích yêu cầu
│   ├── bao_cao_tong_hop.md            # Báo cáo tổng hợp cuối cùng
│   ├── big_picture_overview.md        # Tổng quan: bài toán → RL → mjlab
│   ├── codebase_analysis.md           # Kiến trúc Manager-Based
│   ├── ppo_training_deep_dive.md      # Thuật toán PPO chi tiết
│   ├── reward_engineering.md          # Reward design analysis
│   ├── engineai_integration_design.md # Thiết kế tích hợp PM_v2
│   ├── motion_pipeline.md             # PKL → CSV → NPZ pipeline
│   ├── pmv2_config_analysis.md        # PM_v2 config analysis
│   ├── g1_velocity_evaluation_report.md  # G1 training report
│   └── g1_vs_engineai_comparison.md   # So sánh kỹ thuật
│
├── 📂 code/                           ← Source code
│   ├── asset_zoo/engineai_pmv2/       # Robot MJCF model + 25 mesh STLs
│   ├── tasks/velocity/config/         # Velocity task config
│   ├── tasks/tracking/config/         # Tracking task config (mimic)
│   ├── rl/                            # PPO runner + config
│   └── scripts/                       # train.py, play.py, csv_to_npz.py
│
├── 📂 checkpoints/                    ← Trained models
│   ├── velocity_model_19999.pt        # Velocity policy (20K iters)
│   ├── boxing_model_29999.pt          # Boxing policy (30K iters)
│   ├── kicking_model_19999.pt         # Kicking policy (20K iters)
│   ├── dance_model_4999.pt            # Dance policy (5K iters, A100)
│   └── dance_policy.onnx             # Dance ONNX export
│
├── 📂 videos/                         ← Demo videos (1080p)
│   ├── velocity_1080p.mp4
│   ├── boxing_1080p_60s.mp4
│   ├── kicking_1080p_60s.mp4
│   └── dance_1080p_60s.mp4
│
├── 📂 motions/                        ← Motion reference data (NPZ)
│   ├── pmv2-boxing.npz               # 268KB, 3.2s
│   ├── pmv2-dance.npz                # 698KB, 8.4s
│   └── pmv2-kicking.npz              # 382KB, 4.6s
│
├── 📂 configs/                        ← Full YAML training configs
└── 📂 colab_scripts/                  ← Google Colab training scripts
```

---

## 🔬 Motion Data Pipeline

```
Retarget Mocap → PKL → pkl_to_csv.py → CSV → csv_to_npz.py (MuJoCo FK) → NPZ → Training
```

| Motion | Frames (30fps) | Output (50fps) | Duration | Size |
|:------:|:--------------:|:--------------:|:--------:|:----:|
| Boxing | 97 | ~162 | 3.2s | 268KB |
| Dance | 252 | ~420 | 8.4s | 698KB |
| Kicking | 152 | ~253 | 4.6s | 382KB |

> ⚠️ **Quaternion convention**: PKL dùng `[x,y,z,w]` (scalar-last), MuJoCo dùng `[w,x,y,z]` (scalar-first). Script tự swap.

---

## 🆚 So Sánh G1 vs EngineAI PM_v2

| Dimension | Unitree G1 | EngineAI PM_v2 |
|:----------|:----------:|:--------------:|
| **DOFs** | 29 | 24 |
| **Velocity reward** | 34.16 | **65.98** |
| **Fall rate** | **0%** | 4% |
| **Training speed** | 14.5K steps/s | **80K steps/s** |
| **Mimic motions** | — | 3 (boxing/dance/kick) |

**Key insight**: PM_v2 đạt reward velocity cao hơn G1 gần **2×** nhờ trọng tâm thấp → dễ balance hơn.

---

## 🚀 Quick Start

### Training (trên Google Colab A100)
```bash
# 1. Upload colab_scripts/ lên Google Drive
# 2. Trên Colab notebook:
!python setup_colab.py
!bash train_and_render_dance.sh
```

### Render video từ checkpoint
```bash
cd mjlab
MUJOCO_GL=egl uv run python render_headless.py
```

---

## 📝 Tài Liệu Chi Tiết

Xem thư mục [`docs/`](docs/) để đọc phân tích chi tiết:

| Đọc trước | Tài liệu | Nội dung |
|:----------:|:---------|:---------|
| 1️⃣ | [de_bai_phan_tich.md](docs/de_bai_phan_tich.md) | Phân tích đề bài + chiến lược |
| 2️⃣ | [big_picture_overview.md](docs/big_picture_overview.md) | Tổng quan RL → Simulation → mjlab |
| 3️⃣ | [codebase_analysis.md](docs/codebase_analysis.md) | Kiến trúc Manager-Based chi tiết |
| 4️⃣ | [ppo_training_deep_dive.md](docs/ppo_training_deep_dive.md) | PPO algorithm + hyperparameters |
| 5️⃣ | [reward_engineering.md](docs/reward_engineering.md) | 14 reward terms + design philosophy |
| 6️⃣ | [motion_pipeline.md](docs/motion_pipeline.md) | PKL → CSV → NPZ pipeline |
| 7️⃣ | [engineai_integration_design.md](docs/engineai_integration_design.md) | Thiết kế tích hợp PM_v2 |
| 8️⃣ | [bao_cao_tong_hop.md](docs/bao_cao_tong_hop.md) | **📋 Báo cáo tổng hợp cuối cùng** |

---

**Ứng viên**: Lê Đức Nguyên  
**Framework**: [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo + Warp GPU)  
**Training**: RTX 3090 (server) + A100 80GB (Google Colab)

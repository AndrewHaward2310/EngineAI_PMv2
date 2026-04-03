# 🤖 EngineAI PM_v2 — Humanoid Robot RL Training

> **Integration of EngineAI PM_v2 (24 DOF) humanoid robot into the mjlab framework (MuJoCo + Warp GPU) for locomotion and motion mimicry training.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Warp%20GPU-green)](https://mujoco.org)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-orange)](https://arxiv.org/abs/1707.06347)
[![License](https://img.shields.io/badge/License-Private-red)]()

---

## 📊 Training Results

| Task | Iterations | Envs | GPU | Mean Reward | Time |
|:-----|:----------:|:----:|:---:|:-----------:|:----:|
| **Velocity** (walking) | 20,000 | 4,096 | RTX 3090 | **65.98** | ~7h |
| **Boxing** (punching) | 30,000 | 4,096 | RTX 3090 | **~35** | ~14h |
| **Kicking** | 20,000 | 4,096 | RTX 3090 | **~32** | ~7h |
| **Dance** | 5,000 | 16,384 | A100 80GB | **38.27** | ~4.3h |

---

## 🎥 Demo Videos

### Velocity — Locomotion (velocity command tracking)

https://github.com/user-attachments/assets/a5ed1b38-dc4e-4f33-a737-0bcccb9369ff

### Boxing — Motion Mimicry

https://github.com/user-attachments/assets/63c0d6e3-4d33-439a-a5a2-b85f6bce5aa4

### Kicking — Motion Mimicry

https://github.com/user-attachments/assets/1cc33173-1a33-416c-a7b3-a6432cddf466

### Dance — Motion Mimicry (trained on A100 80GB)

https://github.com/user-attachments/assets/622c792e-5dd5-4f29-af4f-75beae3835d7

---

## 🏗️ Technical Architecture

### Robot: EngineAI PM_v2

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

## 📁 Repository Structure

```
EngineAI_PMv2/
├── 📄 README.md                       ← You are here
│
├── 📂 docs/                           ← Technical documentation (11 files, ~3,730 lines)
│   ├── de_bai_phan_tich.md            # Assignment analysis & strategy
│   ├── bao_cao_tong_hop.md            # Final summary report
│   ├── big_picture_overview.md        # Big picture: RL → Simulation → mjlab
│   ├── codebase_analysis.md           # Manager-Based architecture deep dive
│   ├── ppo_training_deep_dive.md      # PPO algorithm & hyperparameters
│   ├── reward_engineering.md          # Reward design (14 terms)
│   ├── engineai_integration_design.md # PM_v2 integration design
│   ├── motion_pipeline.md             # PKL → CSV → NPZ pipeline
│   ├── pmv2_config_analysis.md        # PM_v2 config analysis
│   ├── g1_velocity_evaluation_report.md  # G1 training report
│   └── g1_vs_engineai_comparison.md   # G1 vs EngineAI comparison
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

| Motion | Input (30fps) | Output (50fps) | Duration | Size |
|:------:|:------------:|:--------------:|:--------:|:----:|
| Boxing | 97 frames | ~162 frames | 3.2s | 268KB |
| Dance | 252 frames | ~420 frames | 8.4s | 698KB |
| Kicking | 152 frames | ~253 frames | 4.6s | 382KB |

> ⚠️ **Quaternion convention**: Input PKL uses `[x,y,z,w]` (scalar-last), MuJoCo uses `[w,x,y,z]` (scalar-first). The conversion script handles the swap automatically.

---

## 🆚 G1 vs EngineAI PM_v2 Comparison

| Dimension | Unitree G1 | EngineAI PM_v2 |
|:----------|:----------:|:--------------:|
| **DOFs** | 29 | 24 |
| **Velocity reward** | 34.16 | **65.98** |
| **Fall rate** | **0%** | 4% |
| **Training speed** | 14.5K steps/s | **80K steps/s** |
| **Mimic motions** | — | 3 (boxing/dance/kick) |

**Key insight**: PM_v2 achieves nearly **2× higher** velocity reward than G1, primarily due to its lower center of mass making balance easier.

---

## 🚀 Quick Start

### Training (on Google Colab A100)
```bash
# 1. Upload colab_scripts/ to Google Drive
# 2. In Colab notebook:
!python setup_colab.py
!bash train_and_render_dance.sh
```

### Render video from checkpoint
```bash
cd mjlab
MUJOCO_GL=egl uv run python render_headless.py
```

---

## 📝 Documentation

See the [`docs/`](docs/) directory for detailed technical analysis:

| Order | Document | Content |
|:-----:|:---------|:--------|
| 1️⃣ | [de_bai_phan_tich.md](docs/de_bai_phan_tich.md) | Assignment analysis & execution strategy |
| 2️⃣ | [big_picture_overview.md](docs/big_picture_overview.md) | Big picture: RL → Simulation → mjlab |
| 3️⃣ | [codebase_analysis.md](docs/codebase_analysis.md) | Manager-Based architecture deep dive |
| 4️⃣ | [ppo_training_deep_dive.md](docs/ppo_training_deep_dive.md) | PPO algorithm + hyperparameters |
| 5️⃣ | [reward_engineering.md](docs/reward_engineering.md) | 14 reward terms + design philosophy |
| 6️⃣ | [motion_pipeline.md](docs/motion_pipeline.md) | PKL → CSV → NPZ conversion pipeline |
| 7️⃣ | [engineai_integration_design.md](docs/engineai_integration_design.md) | PM_v2 integration design |
| 8️⃣ | [bao_cao_tong_hop.md](docs/bao_cao_tong_hop.md) | **📋 Final summary report** |

---

**Candidate**: Le Duc Nguyen  
**Framework**: [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo + Warp GPU)  
**Training**: RTX 3090 (server) + A100 80GB (Google Colab)

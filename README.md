<div align="center">

# 🤖 EngineAI PM_v2

### Humanoid Robot Reinforcement Learning

**24-DOF bipedal robot trained for locomotion and motion mimicry using PPO on MuJoCo Warp GPU**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Warp_GPU-4CAF50)](https://mujoco.org)
[![PPO](https://img.shields.io/badge/PPO-RSL--RL-FF6F00)](https://arxiv.org/abs/1707.06347)
[![Docs](https://img.shields.io/badge/Docs-11_files-8B5CF6)](docs/)

</div>

---

## Results

| Task | Iters | Envs | GPU | Reward | Time |
|:-----|------:|-----:|:---:|-------:|-----:|
| **Velocity** — walking | 20K | 4,096 | RTX 3090 | **65.98** | 7h |
| **Boxing** — punching mimic | 30K | 4,096 | RTX 3090 | ~35 | 14h |
| **Kicking** — kick mimic | 20K | 4,096 | RTX 3090 | ~32 | 7h |
| **Dance** — dance mimic | 5K | 16,384 | A100 80GB | **38.27** | 4.3h |

---

## Demo

#### Velocity — Locomotion

https://github.com/user-attachments/assets/a5ed1b38-dc4e-4f33-a737-0bcccb9369ff

#### Boxing — Motion Mimicry

https://github.com/user-attachments/assets/63c0d6e3-4d33-439a-a5a2-b85f6bce5aa4

#### Kicking — Motion Mimicry

https://github.com/user-attachments/assets/1cc33173-1a33-416c-a7b3-a6432cddf466

#### Dance — Motion Mimicry (A100 80GB)

https://github.com/user-attachments/assets/622c792e-5dd5-4f29-af4f-75beae3835d7

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         ManagerBasedRlEnv            │
                    │                                     │
                    │  Action ─── Obs ─── Reward (14)     │
                    │     │        │        │              │
                    │     ▼        ▼        ▼              │
                    │  ┌─────────────────────────────┐    │
                    │  │  MuJoCo Warp · GPU · CUDA   │    │
                    │  │  N robots × physics @ 200Hz │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  Terminal ── Event ── Curriculum     │
                    └──────────────┬──────────────────────┘
                                   │
                          RslRlVecEnvWrapper
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  PPO · Actor [512,256,128] · ELU    │
                    │  GAE(γ=0.99, λ=0.95) · clip=0.2    │
                    │  Adaptive LR · KL-divergence gate   │
                    └─────────────────────────────────────┘
```

| Parameter | Value |
|:----------|:------|
| Physics timestep | 0.005s (200Hz) |
| Decimation | 4 → control at 50Hz |
| Actor input | 135D (obs) → 24D (actions) |
| Critic input | 261D (privileged obs) → 1D (value) |

---

## Robot

```
EngineAI PM_v2 · 24 DOF
├── Legs (12)    6/leg: hip yaw·roll·pitch, knee, ankle roll·pitch
├── Arms (10)    5/arm: shoulder pitch·roll·yaw, elbow pitch·yaw
├── Waist (1)    yaw
└── Head (1)     yaw
```

---

## Motion Pipeline

```
Mocap → PKL → pkl_to_csv.py → CSV → csv_to_npz.py (MuJoCo FK) → NPZ → RL Training
                                         ↑
                              quaternion swap: [x,y,z,w] → [w,x,y,z]
```

| Motion | Frames | Duration | Size |
|:------:|-------:|---------:|-----:|
| Boxing | 162 @ 50fps | 3.2s | 268KB |
| Dance | 420 @ 50fps | 8.4s | 698KB |
| Kicking | 253 @ 50fps | 4.6s | 382KB |

---

## G1 vs PM_v2

| | G1 | PM_v2 |
|:--|:--:|:-----:|
| DOFs | 29 | **24** |
| Velocity reward | 34.16 | **65.98** |
| Fall rate | **0%** | 4% |
| Throughput | 14.5K steps/s | **80K steps/s** |
| Mimic tasks | — | 3 |

> PM_v2 achieves **2× higher** velocity reward due to lower center of mass.

---

## Project Structure

```
├── docs/           11 technical documents (~3,730 lines)
├── code/           robot MJCF, task configs, RL runner, scripts
├── checkpoints/    4 policies (.pt) + 1 ONNX export
├── videos/         4 demo videos (1080p)
├── motions/        3 NPZ motion files
├── configs/        full YAML training configs
└── colab_scripts/  Google Colab training scripts
```

<details>
<summary><b>📝 Documentation reading order</b></summary>

| # | Document | Topic |
|:-:|:---------|:------|
| 1 | [de_bai_phan_tich.md](docs/de_bai_phan_tich.md) | Assignment analysis & strategy |
| 2 | [big_picture_overview.md](docs/big_picture_overview.md) | RL → Simulation → mjlab overview |
| 3 | [codebase_analysis.md](docs/codebase_analysis.md) | Manager-Based architecture |
| 4 | [ppo_training_deep_dive.md](docs/ppo_training_deep_dive.md) | PPO algorithm & hyperparameters |
| 5 | [reward_engineering.md](docs/reward_engineering.md) | 14 reward terms design |
| 6 | [motion_pipeline.md](docs/motion_pipeline.md) | PKL → CSV → NPZ pipeline |
| 7 | [engineai_integration_design.md](docs/engineai_integration_design.md) | PM_v2 integration |
| 8 | [bao_cao_tong_hop.md](docs/bao_cao_tong_hop.md) | Final summary report |

</details>

<details>
<summary><b>🚀 Quick start</b></summary>

### Train on Google Colab (A100)
```bash
!python setup_colab.py
!bash train_and_render_dance.sh
```

### Render from checkpoint
```bash
MUJOCO_GL=egl uv run python render_headless.py
```

</details>

---

<div align="center">

**Le Duc Nguyen** · [mjlab](https://github.com/mujocolab/mjlab) · RTX 3090 + A100 80GB

</div>

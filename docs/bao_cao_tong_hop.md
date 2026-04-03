# Báo Cáo Tổng Hợp — EngineAI PM_v2 × mjlab

**Ứng viên**: Lê Đức Nguyên  
**Ngày hoàn thành**: 31/03/2026  
**Repository**: [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab)

---

## Tóm Tắt Kết Quả

Bài test yêu cầu phân tích codebase mjlab (Phần A) và train locomotion + mimic policies cho Unitree G1 và EngineAI PM_v2 (Phần B). Kết quả đạt được:

| Hạng mục | Kết quả | Trạng thái |
|----------|---------|-----------|
| **Phân tích codebase** | 10 tài liệu, ~3,500 dòng | ✅ Hoàn thành |
| **G1 velocity** | 30K iters, reward=34.16 | ✅ Hoàn thành |
| **EngineAI velocity** | 20K iters, reward=65.98 | ✅ Hoàn thành |
| **EngineAI boxing** | 30K iters, reward=~35 | ✅ Hoàn thành |
| **EngineAI kicking** | 20K iters, reward=~32 | ✅ Hoàn thành |
| **EngineAI dance** | 5K iters, reward=38.27 | ✅ Hoàn thành |
| **Demo videos** | 4 videos 1080p | ✅ Hoàn thành |
| **So sánh G1 vs EngineAI** | Tài liệu kỹ thuật | ✅ Hoàn thành |

---

## Phần A — Phân Tích Codebase

### Tài liệu đã viết

| # | Tài liệu | Dòng | Nội dung chính |
|---|----------|------|----------------|
| 1 | `de_bai_phan_tich.md` | ~150 | Đề bài gốc + phân tích yêu cầu + chiến lược thực hiện |
| 2 | `big_picture_overview.md` | 600 | Tổng quan: bài toán → RL → simulation → mjlab → pipeline |
| 3 | `codebase_analysis.md` | 579 | Kiến trúc Manager-Based, 7 tầng, data flow chi tiết |
| 4 | `ppo_training_deep_dive.md` | 512 | PPO algorithm, RSL-RL implementation, hyperparameter analysis |
| 5 | `reward_engineering.md` | 398 | Velocity vs tracking reward design, 14 reward terms |
| 6 | `engineai_integration_design.md` | 348 | Thiết kế tích hợp PM_v2: MJCF, constants, config |
| 7 | `motion_pipeline.md` | 330 | PKL → CSV → NPZ pipeline, quaternion convention |
| 8 | `pmv2_config_analysis.md` | 214 | Phân tích config PM_v2 chi tiết |
| 9 | `g1_velocity_evaluation_report.md` | ~120 | G1 evaluation report |
| 10 | `g1_vs_engineai_comparison.md` | ~200 | So sánh kỹ thuật + kết quả training |

### Tóm tắt kiến trúc mjlab

mjlab sử dụng **Manager-Based Architecture** (fork từ NVIDIA IsaacLab) với MuJoCo Warp (GPU) thay cho PhysX:

```
┌─────────────────────────────────────────────────┐
│              ManagerBasedRlEnv                   │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Action   │ │ Obs      │ │ Reward           │ │
│  │ Manager  │ │ Manager  │ │ Manager          │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────────────┘ │
│       │            │            │                │
│  ┌────▼────────────▼────────────▼──────────────┐ │
│  │              Scene (MuJoCo Warp)             │ │
│  │   N robots × physics step on GPU (CUDA)      │ │
│  └──────────────────────────────────────────────┘ │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Terminal │ │ Event    │ │ Curriculum       │ │
│  │ Manager  │ │ Manager  │ │ Manager          │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
                       ▲
                       │ RslRlVecEnvWrapper
                       ▼
              ┌──────────────────┐
              │ MjlabOnPolicy    │
              │ Runner (PPO)     │
              │ via RSL-RL       │
              └──────────────────┘
```

**Điểm mạnh thiết kế:**
- **Separation of concerns**: Thêm robot mới chỉ cần thêm files trong `asset_zoo/` + `tasks/config/` — không đụng core logic
- **GPU-native**: MuJoCo Warp chạy physics batched trên GPU → 80K+ steps/s
- **Config-driven**: Toàn bộ task definition qua Python dataclass — không hardcode

### Thuật toán PPO (tóm tắt)

PPO (Proximal Policy Optimization) cập nhật policy từ từ qua clipped objective:

```
L = -E[ min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A) ]

r(θ) = π_new(a|s) / π_old(a|s)    # probability ratio
A = advantage (GAE, λ=0.95)        # how good is action vs average
ε = 0.2                            # clip range
```

**Đặc biệt trong mjlab:**
- **Adaptive LR**: KL divergence > 2×target → giảm lr; KL < target/2 → tăng lr
- **EmpiricalNormalization**: Running mean/std cho observation → training ổn định
- **ELU activation**: Gradient ≠ 0 everywhere → tốt hơn ReLU cho continuous control

---

## Phần B — Thực Hành

### Quá trình tích hợp EngineAI PM_v2

#### Bước 1: Asset Zoo (Robot Model)

Tạo `asset_zoo/robots/engineai_pmv2/`:
- **`pmv2_constants.py`**: 24 joint names, default pose (từ MJCF keyframe), PD gains (tune empirically)
- **`xmls/pmv2.xml`**: Chuyển đổi từ EngineAI ROS2 workspace MJCF — thêm foot sites, fix actuator params
- **`xmls/meshes/`**: 25 STL mesh files cho rendering

#### Bước 2: Task Configs

**Velocity** (`tasks/velocity/config/engineai_pmv2/`):
- Copy pattern từ G1 → adapt: joint names, base height (0.79m), PD gains
- Debug: air_time reward=2.0 gây nhảy → đặt 0.0

**Tracking** (`tasks/tracking/config/engineai_pmv2/`):
- 9 reward terms: 6 motion tracking + 3 regularization
- Observation: motion command 48D + body state → 135D actor, 261D critic

#### Bước 3: Motion Data Pipeline

```
PKL (giám khảo) → pkl_to_csv.py → CSV → mjlab csv_to_npz → NPZ → WandB → Training
```

- Giải quyết quaternion convention: PKL dùng `[x,y,z,w]`, MuJoCo dùng `[w,x,y,z]`
- 3 NPZ files: boxing (268KB), dance (698KB), kicking (382KB)
- Upload lên WandB registry tự động

#### Bước 4: Training (Multi-GPU)

| Motion | GPU | Envs | Iters | Time | Reward |
|--------|-----|------|-------|------|--------|
| Velocity | RTX 3090 | 4,096 | 20K | ~7h | 65.98 |
| Boxing | RTX 3090 | 4,096 | 30K | ~14h | ~35 |
| Kicking | RTX 3090 | 4,096 | 20K | ~7h | ~32 |
| **Dance** | **A100 80GB** | **16,384** | **5K** | **4.3h** | **38.27** |

**Chiến lược song song**: Boxing/kicking train đồng thời trên server (byobu sessions), dance train trên Google Colab → tổng thời gian giảm ~50%.

### Kết quả G1 Velocity (Baseline)

| Metric | Giá trị |
|--------|---------|
| Task | Mjlab-Velocity-Flat-Unitree-G1 |
| GPU | RTX 3050 Ti (4GB, 512 envs) |
| Iterations | 30,000 |
| Training time | 7h 20m |
| Final reward | 34.16 |
| Fall rate | **0%** |
| Episode length | 987/1000 |

### Debug Log (các vấn đề đã giải quyết)

| # | Vấn đề | Triệu chứng | Nguyên nhân | Fix |
|---|--------|-------------|-------------|-----|
| 1 | Robot rung | Đứng run liên tục | kd quá cao (5→2) | Giảm kd |
| 2 | Robot nhảy | 2 chân nhấc cùng lúc | air_time reward=2.0 | Đặt =0.0 |
| 3 | PD không match | Torque sai tần số | 5Hz vs 10Hz control | Fix frequency |
| 4 | CSV FK sai | Motion hiển thị lệch | Quaternion convention sai | `[x,y,z,w]` → `[w,x,y,z]` |
| 5 | Render treo | Viser blocking loop | Headless Colab | Viết `render_headless.py` |
| 6 | Checkpoint sort | Lấy model_750 thay 4999 | Alphabetical sort | Numeric sort |

---

## Phần C — So Sánh G1 vs EngineAI

| Dimension | G1 | EngineAI PM_v2 | Winner |
|-----------|-----|---------------|--------|
| **DOFs** | 29 | 24 | PM_v2 (simpler) |
| **Balance** | Harder (tall) | Easier (short) | PM_v2 |
| **Velocity reward** | 34.16 | 65.98 | PM_v2 |
| **Fall rate** | 0% | 4% | G1 |
| **Motion flexibility** | 3 DOF waist | 1 DOF waist | G1 |
| **Mimic richness** | (not tested) | 3 motions trained | PM_v2 |
| **Training speed** | 14.5K steps/s | 80K steps/s | PM_v2 (bigger GPU) |

**Key insight**: PM_v2 đạt reward velocity cao hơn G1 gần 2× nhờ trọng tâm thấp → dễ balance. Tuy nhiên PM_v2 vẫn ngã 4% — có thể cải thiện bằng thêm iterations hoặc tune termination threshold.

---

## Danh Sách Deliverables

### Tài liệu (trong `/docs/`)

| File | Nội dung |
|------|----------|
| `de_bai_phan_tich.md` | Đề bài + phân tích chi tiết |
| `big_picture_overview.md` | Tổng quan toàn cảnh |
| `codebase_analysis.md` | Kiến trúc + data flow |
| `ppo_training_deep_dive.md` | Thuật toán PPO chi tiết |
| `reward_engineering.md` | Reward design analysis |
| `engineai_integration_design.md` | Thiết kế tích hợp PM_v2 |
| `motion_pipeline.md` | Motion data pipeline |
| `pmv2_config_analysis.md` | Config analysis |
| `g1_velocity_evaluation_report.md` | G1 training report |
| `g1_vs_engineai_comparison.md` | So sánh kỹ thuật |
| `bao_cao_tong_hop.md` | File này — báo cáo tổng hợp |

### Code (trong `/code/`)

| Thư mục | Nội dung |
|---------|----------|
| `asset_zoo/engineai_pmv2/` | Robot MJCF + constants + 25 mesh STLs |
| `tasks/velocity/config/engineai_pmv2/` | Velocity task config (3 files) |
| `tasks/tracking/config/engineai_pmv2/` | Tracking task config (3 files) |
| `tasks/tracking/rl/` | Custom tracking runner |
| `rl/` | PPO runner + config |
| `scripts/` | train.py, play.py, csv_to_npz.py |

### Trained Models (trong `/checkpoints/`)

| File | Task | Iterations | Size |
|------|------|-----------|------|
| `velocity_model_19999.pt` | Velocity (locomotion) | 20K | 4.9 MB |
| `boxing_model_29999.pt` | Boxing (mimic) | 30K | 6.2 MB |
| `kicking_model_19999.pt` | Kicking (mimic) | 20K | 6.2 MB |
| `dance_model_4999.pt` | Dance (mimic) | 5K | 6.2 MB |
| `dance_policy.onnx` | Dance (ONNX export) | — | 1.3 MB |

### Videos (trong `/videos/`)

| File | Resolution | Duration |
|------|-----------|----------|
| `velocity_1080p.mp4` | 1920×1080 | 6s |
| `boxing_1080p_60s.mp4` | 1920×1080 | 60s |
| `kicking_1080p_60s.mp4` | 1920×1080 | 60s |
| `dance_1080p_60s.mp4` | 1920×1080 | 60s |

### Motion Data (trong `/motions/`)

| File | Frames | Duration |
|------|--------|----------|
| `pmv2-boxing.npz` | 162 (50fps) | 3.2s |
| `pmv2-dance.npz` | 420 (50fps) | 8.4s |
| `pmv2-kicking.npz` | 253 (50fps) | 4.6s |

---

## Kết Luận

Bài test đã hoàn thành đầy đủ hai phần:

**Phần A** — Phân tích codebase mjlab qua 10 tài liệu (~3,500 dòng), bao gồm kiến trúc Manager-Based, thuật toán PPO, reward engineering, motion pipeline, và thiết kế tích hợp robot mới.

**Phần B** — Thực hành train 5 policies:
- 1 G1 velocity (baseline)
- 1 EngineAI velocity + 3 EngineAI mimic (boxing, dance, kicking)

Quá trình tích hợp EngineAI PM_v2 chứng minh khả năng hiểu framework đủ sâu để extend: tạo robot asset, viết task config, convert motion data, debug reward/PD gains, và tối ưu training trên multi-GPU (RTX 3090 + A100).

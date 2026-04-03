# EngineAI PMv2 — Humanoid Robot RL Training Results

## Tổng quan

Bài test tích hợp robot **EngineAI PM_v2** (24 DOF) vào framework **mjlab** (MuJoCo + Warp GPU), bao gồm:
- Cấu hình robot asset (MJCF/XML + mesh STL)
- Pipeline chuyển đổi motion data (PKL → CSV → NPZ)
- Thiết lập môi trường RL cho velocity tracking và motion mimicry
- Huấn luyện policy trên RTX 3090 (server) và A100 80GB (Google Colab)

## Kết quả Training

| Task | Iterations | Envs | Mean Reward | GPU | Thời gian |
|------|-----------|------|-------------|-----|-----------|
| **Velocity** (đi bộ) | 20,000 | 4,096 | 65.98 | RTX 3090 | ~7h |
| **Boxing** (đấm bốc) | 30,000 | 4,096 | ~35 | RTX 3090 | ~14h |
| **Kicking** (đá) | 20,000 | 4,096 | ~32 | RTX 3090 | ~7h |
| **Dance** (nhảy) | 5,000 | 16,384 | **38.27** | A100 80GB | ~4.3h |

## Cấu trúc thư mục

```
EngineAI_PMv2_Submission/
├── README.md                      # File này
├── code/                          # Source code
│   ├── asset_zoo/engineai_pmv2/   # Robot MJCF model + mesh STL
│   ├── tasks/
│   │   ├── velocity/config/       # Cấu hình đi bộ
│   │   └── tracking/config/       # Cấu hình mimic (boxing/dance/kicking)
│   ├── rl/                        # PPO runner + config
│   └── scripts/                   # train.py, play.py, csv_to_npz.py
├── checkpoints/                   # Trained models
│   ├── velocity_model_19999.pt    # Best velocity policy
│   ├── boxing_model_29999.pt      # Best boxing policy
│   ├── kicking_model_19999.pt     # Best kicking policy
│   ├── dance_model_4999.pt        # Best dance policy (A100)
│   └── dance_policy.onnx          # Dance policy (ONNX export)
├── videos/                        # Demo videos (1080p)
│   ├── velocity_1080p.mp4         # Đi bộ
│   ├── boxing_1080p_60s.mp4       # Đấm bốc
│   ├── kicking_1080p_60s.mp4      # Đá
│   └── dance_1080p_60s.mp4        # Nhảy
├── motions/                       # Motion reference data
│   ├── pmv2-boxing.npz
│   ├── pmv2-dance.npz
│   └── pmv2-kicking.npz
├── configs/                       # Full YAML configs
│   ├── velocity_env.yaml
│   ├── boxing_env.yaml
│   └── kicking_env.yaml
├── docs/                          # Tài liệu phân tích (10 files, ~3,500 dòng)
│   ├── de_bai_phan_tich.md        # Đề bài + phân tích chi tiết
│   ├── bao_cao_tong_hop.md        # Báo cáo tổng hợp cuối cùng
│   ├── big_picture_overview.md    # Tổng quan: bài toán → RL → mjlab
│   ├── codebase_analysis.md       # Kiến trúc Manager-Based
│   ├── ppo_training_deep_dive.md  # Thuật toán PPO chi tiết
│   ├── reward_engineering.md      # Reward design analysis
│   ├── engineai_integration_design.md  # Thiết kế tích hợp PM_v2
│   ├── motion_pipeline.md         # PKL → CSV → NPZ pipeline
│   ├── g1_velocity_evaluation_report.md  # G1 training report
│   ├── g1_vs_engineai_comparison.md  # So sánh kỹ thuật
│   └── pmv2_config_analysis.md    # PM_v2 config analysis
└── colab_scripts/                 # Google Colab training scripts
    ├── setup_colab.py
    ├── train_dance.sh
    ├── train_kicking.sh
    ├── train_and_render_dance.sh
    └── render_headless.py
```

## Kiến trúc kỹ thuật

### Robot EngineAI PM_v2
- **24 DOF**: 6 per leg (hip yaw/roll/pitch, knee pitch, ankle roll/pitch) + 4 per arm (shoulder pitch/roll/yaw, elbow pitch) + 2 head
- **Sensors**: IMU (body frame), joint encoders, foot contact
- **Physics**: MuJoCo Warp (GPU-accelerated), CUDA Graph capture

### RL Pipeline
- **Algorithm**: PPO (Proximal Policy Optimization) via rsl_rl
- **Network**: MLP Actor (135→512→256→128→24) + Critic (261→512→256→128→1)
- **Observations**: Motion command (48D) + body state + joint pos/vel + actions
- **Rewards**: Motion tracking (body pos/ori/vel) + regularization (action rate, joint limits, self-collision)

### Tối ưu GPU
- **RTX 3090** (24GB): 4,096 environments, physics trên GPU via MJWarp
- **A100 80GB**: 16,384 environments, giảm iterations tỷ lệ nghịch để giữ tổng data

## Cách chạy lại

### Training (trên Colab A100)
```bash
# 1. Upload colab_scripts/ lên Google Drive
# 2. Trên Colab notebook:
!python /content/drive/MyDrive/humanoid_colab/setup_colab.py
!bash /content/drive/MyDrive/humanoid_colab/train_and_render_dance.sh
```

### Render video từ checkpoint
```bash
cd mjlab
MUJOCO_GL=egl uv run python render_headless.py
```

### Export ONNX (cho deployment)
Policy ONNX đã được export tự động khi save checkpoint. File `dance_policy.onnx` có thể load trực tiếp trên robot thật.

## Motion Data Pipeline
```
Retarget Mocap → PKL → csv_to_npz.py → NPZ → Training
```
- Boxing: 268KB (short motion clip)
- Dance: 698KB (longer choreography)
- Kicking: 382KB (kick sequence)

## Liên hệ
- Training logs đầy đủ trên WandB project: `mjlab`
- Colab notebook backup trên Google Drive

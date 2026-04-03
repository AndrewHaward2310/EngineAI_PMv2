# G1 Velocity Training — Evaluation Report

## Training Summary

| Property | Value |
|---|---|
| **Task** | `Mjlab-Velocity-Flat-Unitree-G1` |
| **Run ID** | `vgn0wcv3` (WandB) |
| **Start** | 2026-03-25 22:50 |
| **End** | 2026-03-26 06:25 |
| **Duration** | 7h 20m 39s |
| **Total Iterations** | 30,000 / 30,000 ✅ |
| **Total Steps** | 368,640,000 |
| **Steps/sec** | ~14,500 |
| **Num Envs** | 512 |
| **Checkpoints** | 601 files (every 50 iters) |
| **Final Model** | `model_29999.pt` (5.3 MB) |
| **ONNX Export** | `2026-03-25_22-50-29.onnx` |
| **GPU** | NVIDIA RTX 3050 Ti (4GB) |
| **Exit Code** | 0 (success) |

## Model Architecture

**Actor** (99 → 29): `Linear(99,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,29)`

**Critic** (111 → 1): `Linear(111,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,1)`

## Reward Progression

```
Iteration |  Mean Reward
----------|-------------
        1 |        -1.35
      100 |        -0.51
      500 |         1.10
    1,000 |         2.40
    2,000 |         7.56
    3,000 |        17.96
    5,000 |        33.89
    6,000 |        28.53
    9,000 |        33.87
   10,000 |        35.62  ← peak
   12,000 |        25.65  ← dip (curriculum change)
   15,000 |        27.91
   18,000 |        29.84
   21,000 |        31.25
   24,000 |        33.27
   27,000 |        35.73
   30,000 |        34.16  ← final
```

> [!NOTE]
> Reward dip around iter 12,000–15,000 is likely due to curriculum expansion (command velocity ranges increasing).

## Final Metrics (iter 29,999)

| Metric | Value |
|---|---|
| **Mean Reward** | 34.16 |
| **Mean Episode Length** | 987.24 steps |
| **Mean Action Std** | 0.47 |
| **Value Loss** | 0.0281 |
| **Surrogate Loss** | -0.0173 |

### Reward Breakdown

| Reward Term | Value | Weight |
|---|---|---|
| Track Linear Velocity | **+1.3483** | 2.0 |
| Track Angular Velocity | +0.3715 | 2.0 |
| Upright | **+0.9694** | 1.0 |
| Pose | +0.6149 | 1.0 |
| Body Angular Vel | -0.0340 | -0.05 |
| Angular Momentum | -0.0406 | -0.02 |
| DOF Pos Limits | -0.0057 | -1.0 |
| Action Rate L2 | **-1.2178** | -0.1 |
| Foot Clearance | -0.1590 | -2.0 |
| Foot Swing Height | -0.0267 | -0.25 |
| Foot Slip | -0.0259 | -0.1 |
| Soft Landing | -0.0006 | -1e-5 |
| Self Collisions | -0.0099 | -1.0 |

### Termination Stats
- **Time out**: 1.17 per episode (normal — reaches max episode length)
- **Fell over**: 0.00 ← robot stays upright

### Velocity Tracking Errors
- **XY velocity error**: 0.9244
- **Yaw velocity error**: 1.8619

### Curriculum (Command Ranges)
| Command | Min | Max |
|---|---|---|
| `lin_vel_x` | -2.0 | 3.0 m/s |
| `lin_vel_y` | -1.0 | 1.0 m/s |
| `ang_vel_z` | -0.7 | 0.7 rad/s |

## Evaluation Video

Video recorded with 300 frames, 4 envs, using `model_29999.pt`:

![G1 Evaluation](/home/asus/.gemini/antigravity/brain/2e743c81-2d42-447a-8565-3c782a1aa004/g1_eval_video.mp4)

**Video path**: [rl-video-step-0.mp4](file:///home/asus/Documents/Humanoid/mjlab/logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/videos/play/rl-video-step-0.mp4)

## Key Files

| File | Path |
|---|---|
| Final checkpoint | [model_29999.pt](file:///home/asus/Documents/Humanoid/mjlab/logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/model_29999.pt) |
| ONNX export | [2026-03-25_22-50-29.onnx](file:///home/asus/Documents/Humanoid/mjlab/logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/2026-03-25_22-50-29.onnx) |
| TensorBoard events | [events.out.tfevents](file:///home/asus/Documents/Humanoid/mjlab/logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/events.out.tfevents.1774453832.asus-ASUS-TUF-Dash-F15-FX517ZE-FX517ZE.1108632.0) |
| WandB output log | [output.log](file:///home/asus/Documents/Humanoid/mjlab/wandb/run-20260325_225033-vgn0wcv3/files/output.log) |
| Eval video | [rl-video-step-0.mp4](file:///home/asus/Documents/Humanoid/mjlab/logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/videos/play/rl-video-step-0.mp4) |

## Evaluation Command Used

```bash
source $HOME/.local/bin/env && cd ~/Documents/Humanoid/mjlab && \
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent trained \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-03-25_22-50-29/model_29999.pt \
  --video True --video-length 300 --num-envs 4
```

> [!TIP]
> The `play` command runs indefinitely by default. The video is saved after `video_length` frames, but the simulation loop continues. Press `Ctrl+C` after the video is recorded, or use a smaller `--video-length` (e.g., 50) for faster results.

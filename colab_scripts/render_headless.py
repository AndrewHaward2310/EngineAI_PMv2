#!/usr/bin/env python3
"""Headless video renderer - no viewer, just render and exit.

Upload to Drive/humanoid_colab/ then run on Colab:
    !cd /content/mjlab && MUJOCO_GL=egl uv run python /content/drive/MyDrive/humanoid_colab/render_headless.py
"""

import os
import sys
import shutil
from pathlib import Path

# Config
DRIVE_DIR = Path("/content/drive/MyDrive/humanoid_colab")
MJLAB_DIR = Path("/content/mjlab")
MOTION_FILE = Path("/content/motions/pmv2-dance.npz")
VIDEO_LENGTH = 3000  # 60s @ 50fps
WIDTH = 1920
HEIGHT = 1080

# Find checkpoint (sort numerically by iteration number)
print(">>> Finding checkpoint...")
ckpt_dir = DRIVE_DIR / "checkpoints"
import re
all_ckpts = list(ckpt_dir.rglob("model_*.pt"))
def get_iter_num(p):
    m = re.search(r'model_(\d+)\.pt', p.name)
    return int(m.group(1)) if m else -1
all_ckpts.sort(key=get_iter_num)
ckpt_file = all_ckpts[-1] if all_ckpts else None

if ckpt_file is None:
    print("ERROR: No checkpoint found!")
    sys.exit(1)
print(f"  Checkpoint: {ckpt_file}")

# Ensure motion file
if not MOTION_FILE.exists():
    os.makedirs(MOTION_FILE.parent, exist_ok=True)
    shutil.copy(DRIVE_DIR / "pmv2-dance.npz", MOTION_FILE)

# Import mjlab
print(">>> Loading environment...")
import torch
import mjlab.tasks  # noqa: register tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from dataclasses import asdict

configure_torch_backends()
device = "cuda:0"
task_id = "Mjlab-Tracking-Flat-EngineAI-PMv2"

# Load configs
env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)

# Set motion file
motion_cmd = env_cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.motion_file = str(MOTION_FILE)

# Set rendering
env_cfg.scene.num_envs = 1
env_cfg.viewer.height = HEIGHT
env_cfg.viewer.width = WIDTH

print(">>> Creating environment...")
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")

# Wrap with video recorder
log_dir = ckpt_file.parent
video_dir = log_dir / "videos" / "play"
print(f"  Video output: {video_dir}")

env = VideoRecorder(
    env,
    video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LENGTH,
    disable_logger=True,
)

# Load policy
print(">>> Loading policy...")
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
runner = runner_cls(env_wrapped, asdict(agent_cfg), device=device)
runner.load(str(ckpt_file), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

# Run simulation (headless, no viewer)
print(f">>> Rendering {VIDEO_LENGTH} frames ({VIDEO_LENGTH/50:.0f}s @ 50fps)...")
print("    This takes 5-10 minutes on A100...")

obs, _ = env_wrapped.reset()
for step in range(VIDEO_LENGTH + 100):  # Extra steps for safety
    actions = policy(obs)
    obs, _, _, _ = env_wrapped.step(actions)
    if (step + 1) % 500 == 0:
        print(f"    Step {step+1}/{VIDEO_LENGTH}")

env.close()
print(">>> Rendering complete!")

# Find output video
video_file = None
for f in sorted(video_dir.glob("*.mp4")):
    video_file = f
    break

if video_file is None:
    # Search broader
    for f in sorted(Path(MJLAB_DIR / "logs").rglob("*.mp4")):
        video_file = f

if video_file and video_file.exists():
    size_mb = video_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Video: {video_file} ({size_mb:.1f} MB)")

    # Pack results
    output_dir = Path("/content/dance_results")
    output_dir.mkdir(exist_ok=True)
    shutil.copy(video_file, output_dir / "dance_1080p_60s.mp4")
    shutil.copy(ckpt_file, output_dir / "dance_model_4999.pt")

    # ONNX
    for onnx in ckpt_file.parent.glob("*.onnx"):
        shutil.copy(onnx, output_dir / onnx.name)

    # Zip
    import zipfile
    zip_path = Path("/content/dance_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.iterdir():
            zf.write(f, f"dance_results/{f.name}")

    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  📦 {zip_path} ({zip_mb:.1f} MB)")

    # Copy to Drive
    drive_video = DRIVE_DIR / "videos"
    drive_video.mkdir(exist_ok=True)
    shutil.copy(video_file, drive_video / "dance_1080p_60s.mp4")
    shutil.copy(zip_path, DRIVE_DIR / "dance_results.zip")
    print(f"  📁 Also saved to Drive/humanoid_colab/videos/")

    print()
    print("=== DONE ===")
    print("Download: Files (📁) → dance_results.zip → ⋮ → Tải xuống")
    print("Or from Drive: humanoid_colab/dance_results.zip")
else:
    print("  ❌ Video file not found!")

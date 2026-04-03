#!/bin/bash
# =============================================================
# Train Dance + Save + Render — ALL IN ONE
# =============================================================
# Chạy trên Colab:
#   !bash /content/drive/MyDrive/humanoid_colab/train_and_render_dance.sh
#
# Script tự động:
#   1. Train dance 2500 iterations (32768 envs, optimized for A100)
#   2. Save checkpoints lên Google Drive (không bao giờ mất)
#   3. Render video 1080p 60s
#   4. Save video lên Google Drive
# =============================================================

set -e
export PATH="$HOME/.local/bin:$PATH"

MJLAB_DIR="/content/mjlab"
DRIVE_DIR="/content/drive/MyDrive/humanoid_colab"
MOTION_FILE="/content/motions/pmv2-dance.npz"

echo "╔════════════════════════════════════════════╗"
echo "║  PMv2 Dance: Train + Save + Render        ║"
echo "╚════════════════════════════════════════════╝"

# === PHASE 1: TRAIN ===
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1/3: Training (5000 iters x 16384 envs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure motion file exists
if [ ! -f "$MOTION_FILE" ]; then
    mkdir -p /content/motions
    cp "$DRIVE_DIR/pmv2-dance.npz" "$MOTION_FILE"
fi

cd "$MJLAB_DIR"
uv run train Mjlab-Tracking-Flat-EngineAI-PMv2 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 16384 \
  --agent.save-interval 250 \
  --agent.num_steps_per_env 16 \
  --agent.algorithm.num_mini_batches 8 \
  --agent.algorithm.num_learning_epochs 3 \
  --env.commands.motion.motion-file "$MOTION_FILE"

echo ""
echo "  ✅ Training complete!"

# === PHASE 2: SAVE CHECKPOINTS TO DRIVE ===
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2/3: Saving checkpoints to Drive"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 "$DRIVE_DIR/save_checkpoints.py"

# Also find the latest checkpoint for rendering
LOG_DIR="$MJLAB_DIR/logs/rsl_rl/pmv2_tracking"
BEST_CKPT=$(find "$LOG_DIR" -name "model_*.pt" 2>/dev/null | sort -t_ -k2 -n | tail -1)

if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: Checkpoint not found after training!"
    exit 1
fi

echo "  Best checkpoint: $(basename $BEST_CKPT)"
echo "  ✅ Checkpoints saved to Drive!"

# === PHASE 3: RENDER VIDEO ===
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3/3: Rendering 1080p 60s video"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$MJLAB_DIR"
MUJOCO_GL=egl uv run src/mjlab/scripts/play.py \
    Mjlab-Tracking-Flat-EngineAI-PMv2 \
    --checkpoint-file "$BEST_CKPT" \
    --motion-file "$MOTION_FILE" \
    --video True \
    --video-length 3000 \
    --video-width 1920 \
    --video-height 1080 \
    --num-envs 1

# Find and verify video
VIDEO_FILE=$(find "$(dirname $BEST_CKPT)" -name "rl-video-step-0.mp4" 2>/dev/null | head -1)
if [ -z "$VIDEO_FILE" ]; then
    VIDEO_FILE=$(find "$LOG_DIR" -name "rl-video-step-0.mp4" 2>/dev/null | sort | tail -1)
fi

if [ -f "$VIDEO_FILE" ]; then
    python3 -c "
import cv2, os
cap = cv2.VideoCapture('$VIDEO_FILE')
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
dur = frames/fps if fps > 0 else 0
cap.release()
mb = os.path.getsize('$VIDEO_FILE')/(1024*1024)
print(f'  Resolution: {w}x{h}')
print(f'  Duration:   {dur:.1f}s ({int(frames)} frames)')
print(f'  Size:       {mb:.1f} MB')
"
    # Save to Drive
    mkdir -p "$DRIVE_DIR/videos"
    cp "$VIDEO_FILE" "$DRIVE_DIR/videos/dance_1080p_60s.mp4"
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  ✅ ALL DONE!                              ║"
    echo "║  📁 Video: Drive/humanoid_colab/videos/    ║"
    echo "║     dance_1080p_60s.mp4                    ║"
    echo "║  💾 Checkpoints: Drive/humanoid_colab/     ║"
    echo "║     checkpoints/                           ║"
    echo "╚════════════════════════════════════════════╝"
else
    echo "⚠️ Video file not found, but checkpoints are saved."
    echo "  You can render later with render_dance.sh"
fi

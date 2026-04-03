#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/mjlab
uv run train Mjlab-Tracking-Flat-EngineAI-PMv2 \
  --agent.max-iterations 20000 \
  --env.scene.num-envs 8192 \
  --agent.save-interval 1000 \
  --env.commands.motion.motion-file /content/motions/pmv2-dance.npz

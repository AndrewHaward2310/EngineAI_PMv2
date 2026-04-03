from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import engineai_pmv2_flat_tracking_env_cfg
from .rl_cfg import engineai_pmv2_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-EngineAI-PMv2",
  env_cfg=engineai_pmv2_flat_tracking_env_cfg(),
  play_env_cfg=engineai_pmv2_flat_tracking_env_cfg(play=True),
  rl_cfg=engineai_pmv2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

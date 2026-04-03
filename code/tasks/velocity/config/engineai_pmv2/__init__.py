from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import engineai_pmv2_flat_env_cfg
from .rl_cfg import engineai_pmv2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-EngineAI-PMv2",
  env_cfg=engineai_pmv2_flat_env_cfg(),
  play_env_cfg=engineai_pmv2_flat_env_cfg(play=True),
  rl_cfg=engineai_pmv2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

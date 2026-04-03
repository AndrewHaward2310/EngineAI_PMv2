"""EngineAI PMv2 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  PMV2_ACTION_SCALE,
  get_pmv2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def engineai_pmv2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PMv2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_pmv2_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PMV2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "LINK_TORSO_YAW"
  motion_cmd.body_names = (
    # Base
    "LINK_BASE",
    # Left leg
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    # Right leg
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    # Torso
    "LINK_TORSO_YAW",
    # Left arm
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    # Right arm
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^.*foot.*$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_TORSO_YAW",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
  )

  cfg.viewer.body_name = "LINK_TORSO_YAW"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg

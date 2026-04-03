"""Velocity environment configurations for EngineAI PM_v2."""

from mjlab.asset_zoo.robots import (
  PMV2_ACTION_SCALE,
  get_pmv2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def engineai_pmv2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM_v2 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_pmv2_robot_cfg()}

  # Set raycast sensor frame to PM_v2 base (like G1 rough sets to pelvis).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "LINK_BASE"

  site_names = ("left_foot", "right_foot")

  # Wire foot height scan to per-foot sites (like G1 rough config does).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove terrain_scan raycast sensor and height_scan observation (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Foot contact sensor.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(LINK_ANKLE_ROLL_L|LINK_ANKLE_ROLL_R)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # Actions.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PMV2_ACTION_SCALE

  # Viewer.
  cfg.viewer.body_name = "LINK_TORSO_YAW"

  # Velocity command visualization.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # Domain randomization targets.
  foot_geom_names = (
    "left_foot_collision",
    "left_foot_toe_collision",
    "right_foot_collision",
    "right_foot_toe_collision",
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  # Variable posture reward — per-joint std adapted for PM_v2 DOFs.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*HIP_PITCH.*": 0.3,
    r".*HIP_ROLL.*": 0.15,
    r".*HIP_YAW.*": 0.15,
    r".*KNEE.*": 0.35,
    r".*ANKLE_PITCH.*": 0.25,
    r".*ANKLE_ROLL.*": 0.1,
    # Waist.
    r".*WAIST_YAW.*": 0.2,
    # Arms.
    r".*SHOULDER_PITCH.*": 0.15,
    r".*SHOULDER_ROLL.*": 0.15,
    r".*SHOULDER_YAW.*": 0.1,
    r".*ELBOW.*": 0.15,
    # Head.
    r".*HEAD_YAW.*": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*HIP_PITCH.*": 0.5,
    r".*HIP_ROLL.*": 0.2,
    r".*HIP_YAW.*": 0.2,
    r".*KNEE.*": 0.6,
    r".*ANKLE_PITCH.*": 0.35,
    r".*ANKLE_ROLL.*": 0.15,
    # Waist.
    r".*WAIST_YAW.*": 0.3,
    # Arms.
    r".*SHOULDER_PITCH.*": 0.5,
    r".*SHOULDER_ROLL.*": 0.2,
    r".*SHOULDER_YAW.*": 0.15,
    r".*ELBOW.*": 0.35,
    # Head.
    r".*HEAD_YAW.*": 0.3,
  }

  # Set body-specific reward targets.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("LINK_TORSO_YAW",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("LINK_TORSO_YAW",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Terminations.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg

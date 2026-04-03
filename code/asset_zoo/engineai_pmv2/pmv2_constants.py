"""EngineAI PM_v2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PMV2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "engineai_pmv2" / "xmls" / "pmv2.xml"
)
assert PMV2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PMV2_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PMV2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# PM_v2 has two motor classes based on MJCF torque limits:
#   - Heavy (164 Nm): hip_pitch, hip_roll, knee_pitch  (armature=0.045325)
#   - Light (61 Nm):  hip_yaw, ankles, waist, shoulders, elbows, head (armature=0.039175)
#
# PD gains are computed from armature using the same physics-based approach as G1:
#   ω_n = 10 × 2π (10 Hz natural frequency)
#   ζ = 2.0 (overdamping ratio)
#   kp = armature × ω_n²
#   kd = 2 × ζ × armature × ω_n
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz — matches G1's proven frequency

# Heavy actuators (164 Nm joints)
ARMATURE_HEAVY = 0.045325
STIFFNESS_HEAVY = ARMATURE_HEAVY * NATURAL_FREQ**2
DAMPING_HEAVY = 2.0 * 2.0 * ARMATURE_HEAVY * NATURAL_FREQ

# Light actuators (61 Nm joints)
ARMATURE_LIGHT = 0.039175
STIFFNESS_LIGHT = ARMATURE_LIGHT * NATURAL_FREQ**2
DAMPING_LIGHT = 2.0 * 2.0 * ARMATURE_LIGHT * NATURAL_FREQ

PMV2_ACTUATOR_HEAVY = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*HIP_PITCH.*",
    ".*HIP_ROLL.*",
    ".*KNEE_PITCH.*",
  ),
  stiffness=STIFFNESS_HEAVY,
  damping=DAMPING_HEAVY,
  effort_limit=164.0,
  armature=ARMATURE_HEAVY,
)

PMV2_ACTUATOR_LIGHT = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*HIP_YAW.*",
    ".*ANKLE.*",
    ".*WAIST_YAW.*",
    ".*SHOULDER.*",
    ".*ELBOW.*",
    ".*HEAD_YAW.*",
  ),
  stiffness=STIFFNESS_LIGHT,
  damping=DAMPING_LIGHT,
  effort_limit=61.0,
  armature=ARMATURE_LIGHT,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.82),
  joint_pos={
    ".*HIP_PITCH.*": 0.0,
    ".*HIP_ROLL.*": 0.0,
    ".*HIP_YAW.*": -0.12,
    ".*KNEE_PITCH.*": 0.24,
    ".*ANKLE_PITCH.*": -0.12,
    ".*ANKLE_ROLL.*": 0.0,
    ".*WAIST_YAW.*": 0.0,
    ".*SHOULDER.*": 0.0,
    ".*ELBOW.*": 0.0,
    ".*HEAD_YAW.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*collision.*", "collision_.*"),
  condim={".*foot.*": 3, ".*": 1},
  priority={".*foot.*": 1},
  friction={".*foot.*": (0.6,)},
)

##
# Final config.
##

PMV2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PMV2_ACTUATOR_HEAVY, PMV2_ACTUATOR_LIGHT),
  soft_joint_pos_limit_factor=0.9,
)


def get_pmv2_robot_cfg() -> EntityCfg:
  """Get a fresh PM_v2 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PMV2_ARTICULATION,
  )


PMV2_ACTION_SCALE: dict[str, float] = {}
for _a in PMV2_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    PMV2_ACTION_SCALE[_n] = 0.25 * _e / _s

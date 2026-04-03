# EngineAI PM_v2 Integration Design — Kế Hoạch Tích Hợp Robot Mới

---

## Kết nối với tài liệu trước

> Bộ tài liệu trước đã giải thích:
> - [codebase_analysis.md](codebase_analysis.md): Manager-based architecture — thêm robot = thêm config
> - [reward_engineering.md](reward_engineering.md): 14 reward terms velocity, 9 terms tracking
> - [motion_pipeline.md](motion_pipeline.md): PKL → CSV → NPZ → WandB
>
> **Doc này** biến kiến thức đó thành **kế hoạch cụ thể** để tích hợp EngineAI PM_v2.

---

## Phần I: Tại Sao Phải Tự Tích Hợp?

G1 đã có sẵn trong mjlab → ai cũng chạy `uv run train` được. **EngineAI PM_v2 không có sẵn** — phải tự:
1. Đọc MJCF model → hiểu robot
2. Tạo asset_zoo entry → nạp robot vào framework
3. Tạo task configs → định nghĩa reward/obs/termination cho robot
4. Adapt motion pipeline → convert PKL data
5. Train + debug → chứng minh framework hoạt động với robot mới

Đây là phần giám khảo đánh giá **hiểu framework đủ sâu để extend**.

---

## Phần II: Phân Tích Robot — EngineAI PM_v2

### 2.1 So sánh specs với G1

| Thông số | G1 | PM_v2 | Ảnh hưởng |
|----------|-----|-------|-----------|
| Tổng DOFs | 29 | 24 | Action/obs space nhỏ hơn |
| Chiều cao base | 0.76m (bent) | 0.82m | Termination threshold khác |
| Chân (mỗi bên) | 6 DOF | 6 DOF | Tương tự |
| Eo | 3 DOF (yaw+roll+pitch) | 1 DOF (yaw only) | Ít flexibility xoay thân |
| Tay (mỗi bên) | 7 DOF | 5 DOF | Ít wrist DOFs |
| Đầu | 0 DOF | 1 DOF (yaw) | Thêm 1 DOF nhỏ |
| Wrist | ✅ 3 DOF | ❌ Không có | Mimic bị giới hạn |

### 2.2 Joint Layout PM_v2 (24 DOFs)

```
Nhóm      │ ID   │ Joint Name         │ Torque (Nm)
──────────┼──────┼────────────────────┼──────────────
Left Leg  │ J00  │ HIP_PITCH_L        │ ±164
          │ J01  │ HIP_ROLL_L         │ ±164
          │ J02  │ HIP_YAW_L          │ ±61
          │ J03  │ KNEE_PITCH_L       │ ±164
          │ J04  │ ANKLE_PITCH_L      │ ±61
          │ J05  │ ANKLE_ROLL_L       │ ±61
──────────┼──────┼────────────────────┼──────────────
Right Leg │ J06  │ HIP_PITCH_R        │ ±164
          │ J07  │ HIP_ROLL_R         │ ±164
          │ J08  │ HIP_YAW_R          │ ±61
          │ J09  │ KNEE_PITCH_R       │ ±164
          │ J10  │ ANKLE_PITCH_R      │ ±61
          │ J11  │ ANKLE_ROLL_R       │ ±61
──────────┼──────┼────────────────────┼──────────────
Waist     │ J12  │ WAIST_YAW          │ ±61
──────────┼──────┼────────────────────┼──────────────
Left Arm  │ J13  │ SHOULDER_PITCH_L   │ ±61
          │ J14  │ SHOULDER_ROLL_L    │ ±61
          │ J15  │ SHOULDER_YAW_L     │ ±61
          │ J16  │ ELBOW_PITCH_L      │ ±61
          │ J17  │ ELBOW_YAW_L        │ ±61
──────────┼──────┼────────────────────┼──────────────
Right Arm │ J18  │ SHOULDER_PITCH_R   │ ±61
          │ J19  │ SHOULDER_ROLL_R    │ ±61
          │ J20  │ SHOULDER_YAW_R     │ ±61
          │ J21  │ ELBOW_PITCH_R      │ ±61
          │ J22  │ ELBOW_YAW_R        │ ±61
──────────┼──────┼────────────────────┼──────────────
Head      │ J23  │ HEAD_YAW           │ ±61
```

### 2.3 Default Pose (từ MJCF keyframe)

```python
# qpos = [x, y, z, qw, qx, qy, qz,     # freejoint (7)
#          J00, J01, ..., J23]            # joint angles (24)

DEFAULT_POS = {
    # Left leg
    ".*HIP_PITCH_L": 0.0,
    ".*HIP_ROLL_L": 0.0,
    ".*HIP_YAW_L": 0.0,
    ".*KNEE_PITCH_L": -0.12,   # Hơi cong gối
    ".*ANKLE_PITCH_L": 0.24,
    ".*ANKLE_ROLL_L": -0.12,
    # Right leg (mirror)
    ".*HIP_PITCH_R": 0.0,
    ".*HIP_ROLL_R": 0.0,
    ".*HIP_YAW_R": 0.0,
    ".*KNEE_PITCH_R": -0.12,
    ".*ANKLE_PITCH_R": 0.24,
    ".*ANKLE_ROLL_R": -0.12,
    # Waist, arms, head
    ".*WAIST_YAW": 0.0,
    ".*SHOULDER_.*": 0.0,
    ".*ELBOW_.*": 0.0,
    ".*HEAD_YAW": 0.0,
}
BASE_HEIGHT = 0.82  # meters
```

### 2.4 Key Body Names

| Role | G1 | PM_v2 |
|------|-----|-------|
| Torso/pelvis | `torso_link` | `LINK_BASE` |
| Left foot | `left_ankle_roll_link` | `LINK_ANKLE_ROLL_L` |
| Right foot | `right_ankle_roll_link` | `LINK_ANKLE_ROLL_R` |
| Foot contact geoms | `left_foot1..7_collision` | Cần xác nhận từ MJCF |

---

## Phần III: Thiết Kế — Files Cần Tạo

### 3.1 Asset Zoo Entry

```
asset_zoo/robots/engineai_pmv2/          ← TẠO MỚI
├── __init__.py                          # Export config
├── pmv2_constants.py                    # Constants, actuators, keyframe
└── xmls/                                # MJCF + meshes
    ├── pm_v2.xml                        # Copy từ EngineAI repo
    └── meshes/                          # Copy mesh files
```

**pmv2_constants.py phải có:**
- `get_spec()` → load MJCF
- Actuator configs (PD gains) → cần compute từ motor specs hoặc tune
- Keyframe (default pose)
- Collision config
- `get_pmv2_robot_cfg()` → EntityCfg
- `PMV2_ACTION_SCALE` → action scaling per joint

### 3.2 Velocity Task Config

```
tasks/velocity/config/engineai_pmv2/     ← TẠO MỚI
├── __init__.py                          # Register task
├── env_cfgs.py                          # Environment config
└── rl_cfg.py                            # PPO hyperparams
```

### 3.3 Tracking Task Config

```
tasks/tracking/config/engineai_pmv2/     ← TẠO MỚI
├── __init__.py                          # Register task
├── env_cfgs.py                          # Environment config
└── rl_cfg.py                            # PPO hyperparams
```

---

## Phần IV: Quyết Định Thiết Kế — Diff Với G1

### 4.1 Config fields — Giữ nguyên vs Thay đổi

| Config Field | G1 Value | PM_v2 Value | Giữ/Đổi | Lý do |
|--------------|----------|-------------|---------|-------|
| **Reward terms** | 14 terms | 14 terms | **Giữ** | Cùng locomotion task |
| **Reward weights** | Hiện tại | Có thể tune | **Giữ trước** | Tune sau nếu cần |
| **PPO hyperparams** | [512,256,128] | [512,256,128] | **Giữ** | Đã proven |
| `num_envs` | 512 | 512 | **Giữ** | GPU 4GB limitation |
| `max_iterations` | 30,000 | 30,000 | **Giữ** | Đủ converge |
| **Robot model** | G1 MJCF | PM_v2 MJCF | **ĐỔI** | Robot khác |
| **Joint names** | 29 G1 joints | 24 PM_v2 joints | **ĐỔI** | Robot khác |
| **Default pose** | G1 keyframe | PM_v2 keyframe | **ĐỔI** | Robot khác |
| **PD gains** | Từ motor specs | Cần compute/tune | **ĐỔI** | Motor khác |
| **Foot bodies** | left/right_ankle_roll_link | LINK_ANKLE_ROLL_L/R | **ĐỔI** | Tên MJCF khác |
| **Torso body** | torso_link | LINK_BASE | **ĐỔI** | Tên MJCF khác |
| **Base height** | 0.76m | 0.82m | **ĐỔI** | Chiều cao khác |
| **Foot geom names** | left_foot1..7_collision | Cần xác nhận | **ĐỔI** | MJCF khác |
| **Variable posture std** | Per-joint G1 | Per-joint PM_v2 | **ĐỔI** | Joints khác |
| **Action scale** | 0.25 × effort/stiffness | Cần compute | **ĐỔI** | Motor specs khác |

### 4.2 PD Gains — Quyết định quan trọng nhất

G1 dùng **motor-spec-based gains**: `kp = reflected_inertia × ω_n²`, `kd = 2 × ζ × reflected_inertia × ω_n`. Đây là approach physics-based, cần biết motor specs.

**Phương án cho PM_v2:**

| Phương án | Ưu | Nhược |
|-----------|-----|-------|
| Copy G1 gains | Nhanh | Motor PM_v2 khác → có thể không ổn |
| Compute từ MJCF actuator specs | Physics-based | Cần hiểu motor model |
| Dùng simple heuristic (kp=100, kd=5) | Dễ | Có thể cần tune nhiều |

**Đề xuất**: Bắt đầu với **simple heuristic** (kp phù hợp torque limit), sanity check, rồi tune nếu cần.

### 4.3 Variable Posture STD — Adapt cho 24 DOFs

```python
# PM_v2 KHÔNG CÓ: waist_roll, waist_pitch, wrist joints
# PM_v2 CÓ THÊM: head_yaw, elbow_yaw

"std_walking": {
    ".*HIP_PITCH.*":     0.30,   # Tương tự G1
    ".*HIP_ROLL.*":      0.15,
    ".*HIP_YAW.*":       0.15,
    ".*KNEE.*":          0.35,
    ".*ANKLE_PITCH.*":   0.25,
    ".*ANKLE_ROLL.*":    0.10,
    ".*WAIST_YAW.*":     0.20,
    ".*SHOULDER_PITCH.*": 0.15,
    ".*SHOULDER_ROLL.*":  0.15,
    ".*SHOULDER_YAW.*":   0.10,
    ".*ELBOW.*":          0.15,
    ".*HEAD_YAW.*":       0.20,
}
```

---

## Phần V: Motion Pipeline Adaptation

### 5.1 csv_to_npz cần sửa

Hiện tại `csv_to_npz.py` hardcode:
- G1 tracking env config (`unitree_g1_flat_tracking_env_cfg`)
- G1 joint names (29 joints)

**Cần**: Tạo version cho PM_v2 hoặc parameterize:
- Dùng PM_v2 tracking env config
- Dùng PM_v2 joint names (24 joints)

### 5.2 PKL → CSV script

```python
# pkl_to_csv.py — đã thiết kế trong motion_pipeline.md
# Key check: quaternion convention
# Key output: [x,y,z, qx,qy,qz,qw, j0,...,j23] per frame
```

### 5.3 Tracking body names cho PM_v2

G1 tracking dùng 14 bodies. PM_v2 cần tương đương:

```python
# G1 bodies         → PM_v2 equivalent
"pelvis"            → "LINK_BASE"
"left_hip_roll_link"   → "LINK_HIP_ROLL_L"
"left_knee_link"       → "LINK_KNEE_PITCH_L"
"left_ankle_roll_link" → "LINK_ANKLE_ROLL_L"
"right_hip_roll_link"  → "LINK_HIP_ROLL_R"
"right_knee_link"      → "LINK_KNEE_PITCH_R"
"right_ankle_roll_link"→ "LINK_ANKLE_ROLL_R"
"torso_link"           → "LINK_BASE"  # PM_v2 ko tách torso
"left_shoulder_roll_link"  → "LINK_SHOULDER_ROLL_L"
"left_elbow_link"      → "LINK_ELBOW_PITCH_L"
# PM_v2 KHÔNG CÓ wrist → bỏ wrist tracking bodies
"right_shoulder_roll_link" → "LINK_SHOULDER_ROLL_R"
"right_elbow_link"     → "LINK_ELBOW_PITCH_R"
```

→ PM_v2 tracking dùng ~12 bodies thay vì 14 (bỏ wrist).

---

## Phần VI: Task Registration

```python
# tasks/velocity/config/engineai_pmv2/__init__.py

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-EngineAI-PMv2",
    env_cfg_fn=engineai_pmv2_flat_env_cfg,
    rl_cfg_fn=engineai_pmv2_ppo_runner_cfg,
)

# tasks/tracking/config/engineai_pmv2/__init__.py

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-EngineAI-PMv2",
    env_cfg_fn=engineai_pmv2_flat_tracking_env_cfg,
    rl_cfg_fn=engineai_pmv2_tracking_ppo_runner_cfg,
)
```

---

## Phần VII: Verification Plan

### Sanity Checks (trước khi train)

```bash
# 1. Zero agent — robot đứng yên, không ngã
uv run play Mjlab-Velocity-Flat-EngineAI-PMv2 --agent zero

# 2. Random agent — robot ngã nhưng không crash
uv run play Mjlab-Velocity-Flat-EngineAI-PMv2 --agent random

# 3. Train help — task đã register đúng
uv run train Mjlab-Velocity-Flat-EngineAI-PMv2 --help
```

**Dấu hiệu đúng:**
- Zero: robot đứng yên ổn định >5s → PD gains OK, default pose OK
- Random: simulation chạy không NaN → MJCF OK, joint limits OK
- Train: không error → task registered, config valid

### Training Quality Checks

| Metric | OK | Cần debug |
|--------|-----|-----------|
| Reward iter 1 | Negative (robot ngã) | NaN → model lỗi |
| Reward iter 1000 | > 0 | Vẫn âm → PD gains sai |
| Episode length | Tăng dần → 1000 | Luôn ngắn → termination quá strict |
| Value loss | Giảm dần | Tăng → critic diverge |

---

## Phần VIII: Dự Đoán Challenges

| Challenge | Likelihood | Mitigation |
|-----------|-----------|------------|
| MJCF mesh path sai | Cao | Copy meshes + fix paths |
| Joint ordering mismatch | Cao | Verify: print robot.joint_names sau load |
| PD gains quá mạnh/yếu | Trung bình | Sanity check zero agent trước |
| Quaternion convention sai | Trung bình | Check identity quaternion |
| Self-collision pattern sai | Thấp | Test with random agent |
| csv_to_npz hardcoded G1 | Chắc chắn | Phải modify/copy script |

---

## Tổng Kết

```
Bước thực hiện:
1. Copy MJCF + meshes vào asset_zoo/
2. Viết pmv2_constants.py (joints, PD gains, keyframe, collision)
3. Viết velocity task config (env_cfgs.py + rl_cfg.py)
4. Register + sanity check (zero/random)
5. Train locomotion
6. Viết tracking task config
7. Adapt csv_to_npz cho PM_v2
8. Convert PKL → CSV → NPZ (3 motions)
9. Train mimic (×3)
10. Record results + so sánh với G1
```

# Phân Tích Chi Tiết G1 vs PMv2 — Bản Đầy Đủ

## Nguồn dữ liệu

Tài liệu này so sánh **3 nguồn** để tìm config đúng cho training:

| # | Nguồn | Đường dẫn | Trạng thái |
|---|-------|-----------|-----------|
| 1 | **G1 config (đã chạy tốt)** | `mjlab/src/.../unitree_g1/` | ✅ Tham chiếu |
| 2 | **Official EngineAI ROS2** | `engineai_ros2_workspace/src/simulation/mujoco/` | ✅ Nguồn gốc |
| 3 | **Mjlab PMv2 config** | `mjlab/src/.../engineai_pmv2/` | ❌ Cần fix |

---

## 1. PD Controller — Bộ điều khiển vị trí khớp

### Công thức PD

$$\tau = k_p \cdot (q_{target} - q_{current}) - k_d \cdot \dot{q}_{current}$$

Trong đó:
- $\tau$ (torque): **Mô-men xoắn** — lực mà motor tạo ra để xoay khớp (đơn vị: Nm)
- $q_{target}$: **Vị trí mục tiêu** — góc mà policy muốn khớp đạt đến (radian)
- $q_{current}$: **Vị trí hiện tại** — góc khớp đang ở (radian)
- $\dot{q}_{current}$: **Vận tốc hiện tại** — tốc độ khớp đang chuyển động (rad/s)
- $k_p$: **Hệ số Proportional** (stiffness) — quyết định **độ cứng** của khớp
- $k_d$: **Hệ số Derivative** (damping) — quyết định **độ giảm chấn** của khớp

**Ví dụ**: Khớp gối đang ở 0° và policy muốn nó đến 10°:
```
τ = 179 × (10° - 0°) - 11.4 × (0 rad/s) = 1790 (lực kéo về 10°)
```
Khi gần đến 10° và đang chuyển động nhanh (5 rad/s):
```
τ = 179 × (10° - 9.5°) - 11.4 × (5) = 89.5 - 57 = 32.5 (giảm tốc)
```

### Cách tính $k_p$ và $k_d$

$$\omega_n = f \times 2\pi$$
$$k_p = I_{reflected} \times \omega_n^2$$
$$k_d = 2 \times \zeta \times I_{reflected} \times \omega_n$$

| Ký hiệu | Tên | Ý nghĩa |
|----------|-----|---------|
| $f$ | Natural frequency | **Tần số dao động tự nhiên** — khớp dao động nhanh cỡ nào. f cao → khớp cứng và phản ứng nhanh |
| $\omega_n$ | Angular frequency | $= f \times 2\pi$ (đổi Hz sang rad/s) |
| $I_{reflected}$ | Reflected inertia (armature) | **Quán tính phản chiếu** = quán tính rotor + hộp giảm tốc. Motor nặng → armature lớn → cần $k_p$ lớn hơn |
| $\zeta$ | Damping ratio | **Tỉ số giảm chấn**. ζ=1: giảm chấn tới hạn. ζ=2: quá giảm chấn (không dao động, phản ứng chậm hơn) |

### Tại sao chia Heavy và Light?

PMv2 có **2 loại motor vật lý** khác nhau (theo datasheet EngineAI):

| Loại | Motor | Khớp sử dụng | Effort (Nm) | Armature |
|------|-------|-------------|:-----------:|:--------:|
| **Heavy** | 164 Nm | HIP_PITCH, HIP_ROLL, KNEE_PITCH | 164 | 0.045325 |
| **Light** | 61 Nm | HIP_YAW, ANKLE, WAIST, SHOULDER, ELBOW, HEAD | 61 | 0.039175 |

Motor Heavy khỏe gấp ~2.7× motor Light → cần $k_p$ riêng cho mỗi loại. G1 có tới 6 loại motor khác nhau.

### Bảng so sánh $k_p$, $k_d$

| Motor | $k_p$ | $k_d$ | Effort | Action Scale |
|-------|:-----:|:-----:|:------:|:------------:|
| G1 hip_pitch/yaw (88Nm) | 177.7 | 11.3 | 88 | 0.124 |
| G1 hip_roll/knee (139Nm) | 256.6 | 16.3 | 139 | 0.135 |
| G1 ankle (2×25Nm) | 43.4 | 2.8 | 50 | 0.288 |
| **PMv2 CŨ (5Hz)** Heavy | 44.7 | 5.7 | 164 | **0.917** ❌ |
| **PMv2 MỚI (10Hz)** Heavy | 178.9 | 11.4 | 164 | **0.229** ✅ |
| **PMv2 MỚI (10Hz)** Light | 154.7 | 9.9 | 61 | **0.099** ✅ |

### Action Scale — Nghĩa là gì?

**Policy neural network** output giá trị $a \in [-1, 1]$.  
**Action scale** chuyển đổi thành **góc quay thực tế**:

$$q_{target} = q_{default} + a \times \text{action\_scale}$$

**Mục đích**: Giới hạn phạm vi di chuyển cho mỗi loại motor. Motor mạnh (effort lớn) nhưng $k_p$ cũng lớn → action scale nhỏ → di chuyển từ từ, ổn định.

**Công thức**:
$$\text{action\_scale} = 0.25 \times \frac{\text{effort\_limit}}{k_p}$$

| Ví dụ | Scale | Policy output=1.0 → khớp di chuyển |
|-------|:-----:|:----:|
| PMv2 CŨ Heavy | **0.917** | ±0.917 rad = **±52.5°** ← quá lớn! |
| PMv2 MỚI Heavy | 0.229 | ±0.229 rad = ±13.1° ← hợp lý |
| G1 hip_roll/knee | 0.135 | ±0.135 rad = ±7.7° |

> [!IMPORTANT]
> Trạng thái: ✅ Đã fix — đổi 5Hz → 10Hz trong [pmv2_constants.py:49](file:///home/asus/Documents/Humanoid/mjlab/src/mjlab/asset_zoo/robots/engineai_pmv2/pmv2_constants.py#L49)

---

## 2. Keyframe — Tư thế đứng ban đầu

### Hệ tọa độ MuJoCo

```
        Z ↑ (chiều cao)
          |
          |    Y → (sang trái)
          |   /
          |  /
          | /
          O -----→ X (hướng trước mặt robot)
       (mặt đất)
```

- **Gốc tọa độ O**: Tâm mặt đất (z=0 = mặt sàn)
- **`pos = (x, y, z)`**: Vị trí **LINK_BASE** (thân chính) của robot so với O
  - `pos = (0, 0, 0.82)` nghĩa là tâm thân robot ở **cao 0.82m so với mặt đất**
- **Tâm robot (LINK_BASE)**: Nằm ở vùng **hông/pelvis** — nơi gắn freejoint

### Keyframe ảnh hưởng đến training thế nào?

1. **Khởi tạo episode**: Robot spawn ở tư thế này → nếu sai → ngã ngay
2. **Điểm tham chiếu reward `pose`**: Robot được thưởng khi giữ gần tư thế default
3. **`use_default_offset=True`**: Target = default_pos + action × scale → nếu default sai → target sai

### Có thể config lại keyframe không?

**Có**, chỉ cần sửa `HOME_KEYFRAME` trong `pmv2_constants.py`. Không cần sửa MJCF XML.
Trong mjlab framework, `EntityCfg.InitialStateCfg` override keyframe XML khi spawn robot.

### So sánh 3 nguồn keyframe

Phát hiện quan trọng: **Keyframe trong mjlab pmv2.xml bị lệch 1 vị trí so với official!**

Keyframe official EngineAI (từ [serial_pm_v2.xml:97](file:///home/asus/Documents/Humanoid/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/robot/pm_v2/xml/serial_pm_v2.xml#L97)):
```
qpos = 0 0 0.82 1 0 0 0 | 0 0 -0.12 0.24 -0.12 0 | 0 0 -0.12 0.24 -0.12 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0
                          ↑     Left leg (đối xứng)  ↑     Right leg (đối xứng)  ↑  Upper body = 0
```

| Khớp | Official EngineAI | Mjlab pmv2.xml | HOME_KEYFRAME code | G1 |
|------|:-----------------:|:--------------:|:------------------:|:--:|
| **HIP_PITCH** | 0 | 0 | -0.15 ❌ | -0.312 |
| **HIP_ROLL** | 0 | 0 | 0 | 0 |
| **HIP_YAW** | **-0.12** | 0 ❌ | 0 ❌ | 0 |
| **KNEE_PITCH** | **+0.24** | -0.12 ❌ | +0.30 ❌ | +0.669 |
| **ANKLE_PITCH** | **-0.12** | +0.24 ❌ | -0.15 ❌ | -0.363 |
| **ANKLE_ROLL** | 0 | -0.12 ❌ | 0 | 0 |
| **Chiều cao** | 0.82m | 0.82m | 0.77m ❌ | 0.76m |

> [!CAUTION]
> Mjlab pmv2.xml keyframe **bị lệch 1 slot** so với official — values `-0.12, 0.24, -0.12` bị dồn sang chỗ sai. Đây là **bug trong file XML** mà mình đang dùng.

### Fix: Dùng keyframe chính thức từ EngineAI

```python
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.82),
  joint_pos={
    ".*HIP_PITCH.*": 0.0,
    ".*HIP_ROLL.*": 0.0,
    ".*HIP_YAW.*": -0.12,       # Official EngineAI
    ".*KNEE_PITCH.*": 0.24,     # +13.8° — gối cong nhẹ
    ".*ANKLE_PITCH.*": -0.12,   # -6.9° — cổ chân bù lại
    ".*ANKLE_ROLL.*": 0.0,
    ".*WAIST_YAW.*": 0.0,
    ".*SHOULDER.*": 0.0,
    ".*ELBOW.*": 0.0,
    ".*HEAD_YAW.*": 0.0,
  },
  joint_vel={".*": 0.0},
)
```

> [!NOTE]
> Tư thế này có ý nghĩa vật lý: gối cong nhẹ (+0.24 rad = +13.8°) + cổ chân bù (-0.12 rad = -6.9°) → robot đứng với trọng tâm hơi thấp → ổn định hơn đứng thẳng hoàn toàn.

---

## 3. Reward — Bảng phân tích đầy đủ

*(Không thay đổi so với bản trước — xem cùng section)*

Chỉ có **1 thay đổi** cần thiết:

```python
cfg.rewards["air_time"].weight = 0.0  # ← 2.0 → 0.0 (giống G1)
```

**Lý do**: G1 flat đã tắt air_time vì `foot_clearance` (weight=-2.0) đã đủ ép robot nâng chân. Thêm air_time=2.0 khiến robot hack reward bằng cách ngồi + giơ chân.

---

## 4. Phát hiện thêm: Joint damping/frictionloss

Official EngineAI XML có `damping=0.6, frictionloss=0.8` trên joints.
Mjlab pmv2.xml có `damping=0, frictionloss=0`.

**Đây là intentional** — mjlab dùng PD controller (kp/kd) thay cho MuJoCo built-in damping. Nếu có cả hai sẽ bị "double damping" → robot quá chậm. G1 cũng đặt damping=0. **Không cần thay đổi.**

---

## Tổng hợp: 3 Fix cần áp dụng

| # | File | Dòng | Thay đổi | Trạng thái |
|---|------|:----:|---------|:---------:|
| 1 | `pmv2_constants.py` | 49 | `NATURAL_FREQ` 5Hz → 10Hz | ✅ Xong |
| 2 | `pmv2_constants.py` | 92-107 | `HOME_KEYFRAME` theo official EngineAI | ⏳ Cần fix |
| 3 | `env_cfgs.py` | 167 | `air_time.weight` 2.0 → 0.0 | ⏳ Cần fix |

### Kế hoạch kiểm tra

**Zero agent test** (action=0 → PD giữ robot đứng yên ở keyframe):
```bash
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-EngineAI-PMv2 \
  --agent zero --video True --video-length 50 --num-envs 1
```
**Kết quả đúng**: Robot đứng thẳng, gối hơi cong, không ngã.

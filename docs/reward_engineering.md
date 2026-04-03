# Reward Engineering — Ngôn Ngữ Giữa Con Người Và Robot

---

## Kết nối với tài liệu trước

> **Doc trước** ([ppo_training_deep_dive.md](ppo_training_deep_dive.md)) giải thích PPO tối ưu hóa **tổng reward**. **Doc này** giải thích tổng reward đó **gồm những gì** và **tại sao thiết kế như vậy.**

Reward function = cách duy nhất để "nói" cho robot biết bạn muốn gì. Robot chỉ **tối đa hóa con số** — nó không hiểu "đi đẹp" hay "tự nhiên". Mọi hành vi mong muốn phải được **mã hóa thành reward terms**.

---

## Phần I: Triết Lý Thiết Kế Reward

### 1.1 Positive + Penalty = Hành vi mong muốn

```
Total Reward = Σ(Positive terms) + Σ(Penalty terms)
                ↑ khuyến khích         ↑ ngăn cản
                hành vi tốt           hành vi xấu
```

**Nếu chỉ có positive** (thưởng đi đúng tốc độ):
- Robot tìm cách "cheat" — ví dụ: lắc qua lại ngay tại chỗ cũng tạo velocity
- Hoặc tìm pose lạ có reward cao mà không thực sự đi

**Nếu quá nhiều penalty** (phạt mọi thứ):
- Robot đứng yên (an toàn nhất) — mọi penalty = 0
- Reward tổng = 0, nhưng robot không ngã → "tối ưu" theo góc nhìn robot

→ **Cân bằng** positive/penalty là kỹ thuật quan trọng nhất.

### 1.2 Exp Kernel vs L2 — Chọn hàm nào?

Hầu hết reward terms dùng **exponential kernel** thay vì L2 distance:

```python
# L2 (linear penalty):
reward = -||error||²                  # Lớn khi error lớn → gradient quá mạnh

# Exp kernel (bounded reward):
reward = exp(-||error||² / σ²)        # ∈ [0, 1], gradient mượt mà

# σ (std) controls sensitivity:
#   σ nhỏ (0.25) → rất nhạy, chỉ reward khi error ≈ 0
#   σ lớn (1.0)  → dễ dãi, reward kể cả khi error vừa phải
```

**Tại sao exp kernel?** Vì L2 penalty tạo gradient rất lớn khi error lớn → robot "hoảng loạn" → hành vi giật. Exp kernel **bounded** [0,1] → gradient mượt mà → hành vi tự nhiên.

---

## Phần II: Velocity Task — 14 Reward Terms Chi Tiết

### 2.1 Tổng quan cấu trúc

```
                                   ┌── track_linear_velocity  (+2.0)
                    POSITIVE       ├── track_angular_velocity (+2.0)
                    (mục tiêu)     ├── upright                (+1.0)
                                   └── pose                   (+1.0)
                                   
14 Reward Terms ──►
                                   ┌── action_rate_l2         (-0.05)
                                   ├── dof_pos_limits         (-1.0)
                    PENALTY        ├── foot_clearance         (-2.0)
                    (regularize)   ├── foot_swing_height      (-0.25)
                                   ├── foot_slip              (-0.1)
                                   ├── soft_landing           (-1e-5)
                                   ├── self_collisions        (-1.0)
                                   ├── body_ang_vel           (-0.05)
                                   └── angular_momentum       (-0.02)
```

### 2.2 Nhóm Positive — Mục tiêu chính

#### `track_linear_velocity` (weight = +2.0) ⭐

```python
reward = exp(-||v_actual_xy - v_command_xy||² / σ²)
# σ = √0.25 = 0.5

# Ví dụ: command = [1.0, 0.0] m/s (đi thẳng tới)
#   actual = [0.95, 0.02] → error = 0.055 → reward ≈ 0.80 ✅
#   actual = [0.3, 0.0]  → error = 0.49   → reward ≈ 0.14 ❌
#   actual = [0.0, 0.0]  → error = 1.0    → reward ≈ 0.02 ❌
```

**Tại sao weight=2.0** (cao nhất)? Vì đây là **mục tiêu chính** — robot phải đi đúng tốc độ lệnh. Mọi thứ khác là phụ trợ.

**Nếu bỏ term này?** Robot không có lý do để di chuyển → đứng yên (penalty = 0 → an toàn nhất).

#### `track_angular_velocity` (weight = +2.0)

Tương tự linear velocity nhưng cho xoay (yaw):
```python
reward = exp(-||ω_z_actual - ω_z_command||² / σ²)
# σ = √0.5 ≈ 0.71 → dễ dãi hơn linear (xoay khó kiểm soát hơn)
```

#### `upright` (weight = +1.0)

```python
# Đo góc giữa trục z của torso và phương thẳng đứng
reward = exp(-||tilt_angle||² / σ²)
# σ = √0.2 ≈ 0.45

# Nghiêng 5°  → reward ≈ 0.97 (gần hoàn hảo)
# Nghiêng 20° → reward ≈ 0.53 (cảnh báo)
# Nghiêng 40° → reward ≈ 0.07 (gần ngã)
```

**Nếu bỏ?** Robot có thể đi bằng cách nghiêng người mạnh → không tự nhiên, dễ ngã trên robot thật.

#### `pose` — Variable Posture (weight = +1.0) ⭐⭐

Đây là term **tinh tế nhất** — reward thay đổi theo tốc độ:

```python
# Tính khoảng cách từ pose hiện tại đến default pose
# std thay đổi theo speed:
speed = ||v_command_xy||

if speed < 0.05:       # Đứng yên
    std = std_standing  # Rất chặt (0.05) → phải gần default
elif speed < 1.5:      # Đi bộ
    std = std_walking   # Lỏng hơn (0.1-0.35)
else:                   # Chạy
    std = std_running   # Rất lỏng (0.2-0.6)

reward = exp(-Σ(joint_error² / std²))
```

**Tại sao cần variable posture?**

| Tốc độ | Hành vi mong muốn | Std |
|--------|-------------------|-----|
| Đứng yên | Giữ pose chuẩn, không rung | Rất chặt (0.05) |
| Đi bộ | Đầu gối cong, tay đung đưa | Lỏng knee=0.35, arm=0.15 |
| Chạy | Đầu gối cong sâu, tay vung mạnh | Rất lỏng knee=0.6, arm=0.35 |

**Nếu dùng 1 std cố định?**
- Std chặt (0.05): robot không dám cong gối → đi cứng, shuffle
- Std lỏng (0.5): robot rung khi đứng yên → không ổn định

**Per-joint std cho G1 (walking mode):**

```python
".*hip_pitch.*":     0.30   # Hip cần swing → lỏng
".*hip_roll.*":      0.15   # Hip roll ít → chặt hơn
".*knee.*":          0.35   # Knee cong nhiều → lỏng nhất
".*ankle_pitch.*":   0.25   # Ankle pitch cho foot clearance
".*ankle_roll.*":    0.10   # Ankle roll quan trọng cho balance → chặt
".*waist_roll.*":    0.08   # Waist roll phải ổn định → rất chặt
".*shoulder_pitch.*": 0.15  # Arm swing tự nhiên
".*wrist.*":         0.30   # Wrist ít ảnh hưởng → lỏng
```

### 2.3 Nhóm Penalty — Regularization

#### `action_rate_l2` (weight = -0.05)

```python
reward = -||action_t - action_{t-1}||²
```

**Ý nghĩa**: Phạt khi action thay đổi đột ngột giữa 2 steps liên tiếp.

**Nếu bỏ?** Robot giật mạnh — motor command nhảy từ -1 lên +1 liên tục. Trên robot thật:
- Motor overheat
- Gear hỏng nhanh
- Tiếng ồn lớn

#### `foot_clearance` (weight = -2.0) — Penalty nặng nhất!

```python
# Phạt khi chân swing không nâng đủ cao
error = max(0, target_height - actual_foot_height)
reward = -error²
# target_height = 0.1m (10cm)
```

**Tại sao weight=-2.0 (nặng nhất)?** Vì foot clearance cực kỳ quan trọng:
- Chân không nâng → kéo lê → **vấp ngã** trên robot thật
- Chỉ active khi robot đang đi (command > 0.05 m/s)

#### `foot_slip` (weight = -0.1)

```python
# Phạt khi chân đang chạm đất mà vẫn trượt
reward = -||v_foot_xy||² × is_contact
```

**Nếu bỏ?** Robot "trượt patin" — chân chạm đất nhưng vẫn di chuyển → không tự nhiên, sim-to-real gap lớn (ma sát thật khác sim).

#### `self_collisions` (weight = -1.0)

```python
# Phạt khi body parts va chạm nhau
reward = -sum(contact_force > threshold)
```

**Nếu bỏ?** Robot có thể tối ưu bằng cách ép tay vào body → force lạ → behavior không transfer sang robot thật.

#### `dof_pos_limits` (weight = -1.0)

```python
# Phạt khi joint gần giới hạn cơ khí
reward = -sum(max(0, q - q_max) + max(0, q_min - q))
```

#### Các penalty nhẹ

| Term | Weight | Ý nghĩa |
|------|--------|---------|
| `body_ang_vel` | -0.05 | Phạt torso lắc → ổn định thân |
| `angular_momentum` | -0.02 | Phạt momentum quay → chuyển động mượt |
| `foot_swing_height` | -0.25 | Phạt chân swing sai chiều cao target |
| `soft_landing` | -1e-5 | Phạt NHẸ đập chân mạnh → khuyến khích đặt chân nhẹ nhàng |
| `air_time` | 0.0 | **Disabled** cho G1 (dùng cho Go1 quadruped) |

---

## Phần III: Tracking Task — So Sánh Với Velocity

### 3.1 Sự khác biệt cốt lõi

| Khía cạnh | Velocity | Tracking |
|-----------|----------|----------|
| **Mục tiêu** | Đi đúng tốc độ lệnh | Bắt chước chuyển động mẫu |
| **Command** | `[v_x, v_y, ω_z]` (3 số) | Motion clip (N frames × body positions) |
| **# Reward terms** | 14 | 9 |
| **Positive terms** | Velocity tracking + posture | 6 motion matching terms |
| **Penalty terms** | 10 (chi tiết gait) | 3 (chỉ regularization cơ bản) |
| **Termination** | Ngã (tilt > 70°) | Sai position/orientation quá nhiều |

### 3.2 Tracking Rewards — 6 Motion Matching Terms

```
Tracking nhắm đến 14 bodies của G1:
  pelvis, left/right hip, knee, ankle (6 điểm chân)
  torso (1), shoulders, elbows, wrists (6 điểm tay)
```

| Term | Weight | Măt | σ |
|------|--------|------|---|
| `motion_global_root_pos` | +0.5 | Vị trí gốc (anchor) so với reference | 0.3 |
| `motion_global_root_ori` | +0.5 | Hướng gốc so với reference | 0.4 |
| `motion_body_pos` | **+1.0** | Vị trí relative các bodies | 0.3 |
| `motion_body_ori` | **+1.0** | Hướng relative các bodies | 0.4 |
| `motion_body_lin_vel` | **+1.0** | Vận tốc tuyến tính các bodies | 1.0 |
| `motion_body_ang_vel` | **+1.0** | Vận tốc góc các bodies | 3.14 |

**Tracking có 3 cấp so khớp:**
1. **Position** (ở đâu): body nào ở tọa độ nào
2. **Orientation** (hướng nào): body xoay góc nào  
3. **Velocity** (nhanh cỡ nào): body di chuyển/xoay nhanh bao nhiêu

→ Tất cả 3 cấp đều quan trọng → weight đều là 1.0.

### 3.3 Tracking Penalties — Ít hơn nhưng mạnh hơn

| Term | Velocity | Tracking | Tại sao khác |
|------|----------|----------|-------------|
| `action_rate_l2` | -0.05 | -0.1 | Tracking dùng -0.1 (chặt hơn 2×) |
| `joint_limit` | -1.0 | **-10.0** | ×10 vì mimic có thể push joints cực đoan |
| `self_collisions` | -1.0 | **-10.0** | ×10 vì boxing/kicking dễ tự va |

**Tại sao tracking penalty ít hơn?**

Velocity cần 10 penalty terms vì robot tự tìm gait — cần nhiều ràng buộc để gait tự nhiên (foot clearance, slip, swing height...).

Tracking đã CÓ motion reference → gait pattern đã implicit trong data. Robot chỉ cần follow → ít cần penalty gait.

Nhưng penalty nặng hơn (×10) vì mimic motions (boxing, kicking) push joints mạnh → cần ràng buộc cứng hơn.

### 3.4 Tracking Observations — Khác biệt then chốt

```
Velocity actor nhận:
  [base_lin_vel, base_ang_vel, projected_gravity,     ← sensor readings
   joint_pos, joint_vel, actions,                     ← robot state
   command (v_x, v_y, ω_z)]                           ← 3 số

Tracking actor nhận:
  [command (motion features),                          ← motion reference target
   motion_anchor_pos_b (relative target position),    ← ĐI ĐÂU
   motion_anchor_ori_b (relative target orientation), ← HƯỚNG NÀO
   base_lin_vel, base_ang_vel,                        ← sensor readings
   joint_pos, joint_vel, actions]                     ← robot state
```

**Tracking THÊM gì?** `motion_anchor_pos_b` và `motion_anchor_ori_b` — cho robot biết "frame tiếp theo bạn phải ở đâu, hướng nào". Velocity chỉ biết tốc độ mục tiêu.

### 3.5 Tracking Termination — Khác hoàn toàn

| | Velocity | Tracking |
|---|----------|----------|
| Điều kiện ngã | Tilt > 70° | — |
| Position quá xa | — | `anchor_pos_z > 0.25m` |
| Orientation sai | — | `anchor_ori_error > 0.8` |
| End-effector sai | — | `ee_pos_z > 0.25m` (4 bodies) |

Tracking dùng **motion-based termination**: nếu robot sai quá xa so với reference → episode fail → reset → thử lại.

---

## Phần IV: Curriculum — Tăng Độ Khó Dần

### 4.1 Velocity Command Curriculum

```python
velocity_stages = [
    {"step": 0,           "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": 5000 × 24,   "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    {"step": 10000 × 24,  "lin_vel_x": (-2.0, 3.0)},
]
```

| Stage | Iter | lin_vel_x range | Ý nghĩa |
|-------|------|-----------------|---------|
| 1 | 0-5K | [-1.0, 1.0] m/s | Đi chậm, học balance |
| 2 | 5K-10K | [-1.5, 2.0] m/s | Mở rộng tốc độ |
| 3 | 10K+ | [-2.0, 3.0] m/s | Toàn bộ range, bao gồm chạy |

**Tại sao cần curriculum?**

Nếu bắt đầu với range [-2, 3] m/s:
- Robot mới (untrained) nhận command 3m/s → không biết chạy → ngã ngay → reward ≈ 0
- Toàn bộ episode chỉ là ngã → PPO không có tín hiệu học

Nếu bắt đầu với [-1, 1] m/s:
- Robot dễ balance hơn ở tốc độ thấp → reward > 0 → PPO học được
- Dần mở rộng range → robot đã biết đi slow, học thêm fast

**Đây giải thích reward dip ở iter 12K-15K**: khi curriculum mở stage 3 (range mở rộng), robot gặp command nhanh hơn chưa biết → reward giảm tạm → rồi hồi phục khi adapt.

### 4.2 Tracking — Không có velocity curriculum

Tracking dùng **RSI (Reference State Initialization)** thay vì command curriculum:
- Mỗi episode, robot bắt đầu từ **random frame** trong motion clip
- Random nhẹ pose/velocity → buộc robot adapt
- Không có stage progression — motion clip đã cố định

---

## Phần V: Nguyên Tắc Thiết Kế Reward

### 5.1 Dense vs Sparse

| Loại | Ví dụ | Ưu | Nhược |
|------|-------|-----|-------|
| **Dense** | exp(-error²/σ²) mỗi step | Robot biết ngay tốt/xấu | Cần thiết kế cẩn thận |
| **Sparse** | +1 khi đến đích, 0 otherwise | Đơn giản | Robot không biết đang tiến bộ |

mjlab dùng **100% dense rewards** — tính mỗi step (0.02s). Tại sao? Vì locomotion cần feedback liên tục — không có "đích" rõ ràng.

### 5.2 Weight Tuning — Nguyên tắc

1. **Positive terms weight tổng > negative terms weight tổng** → robot có motivation di chuyển
2. **Mục tiêu chính weight cao nhất** → velocity tracking = 2.0 (cao nhất)
3. **Penalty quan trọng cho sim-to-real > penalty thẩm mỹ** → foot_clearance = -2.0 > body_ang_vel = -0.05
4. **Bắt đầu penalty nhẹ, tăng nếu cần** → nếu robot trượt quá → tăng foot_slip weight

### 5.3 Common Failure Modes

| Vấn đề | Nguyên nhân reward | Giải pháp |
|--------|-------------------|-----------|
| Robot đứng yên | Penalty quá mạnh so với positive | Tăng velocity tracking weight |
| Robot giật | Thiếu action_rate penalty | Thêm/tăng action_rate_l2 |
| Robot shuffle (không nâng chân) | Thiếu foot_clearance | Thêm foot_clearance penalty |
| Robot "cheat" (lắc tại chỗ) | Velocity tracking dùng L2 thay vì exp | Đổi sang exp kernel |
| Robot ngã khi xoay | ang_vel tracking quá aggressive | Giảm weight hoặc tăng σ |

---

## Phần VI: Tổng Kết — Bảng So Sánh Hoàn Chỉnh

| | Velocity Task | Tracking Task |
|---|---|---|
| **Mục tiêu** | Đi theo lệnh tốc độ | Bắt chước motion clip |
| **Input** | 3 số (vx, vy, ωz) | N frames × M bodies |
| **Positive rewards** | 4 terms (vel + posture) | 6 terms (position + orientation + velocity of bodies) |
| **Penalties** | 10 terms (gait quality) | 3 terms (basic regularization) |
| **Penalty intensity** | Nhẹ (-0.02 đến -2.0) | Mạnh (-0.1 đến -10.0) |
| **Curriculum** | 3 velocity stages | RSI (random frame start) |
| **Termination** | Tilt > 70° | Position/orientation error quá lớn |
| **Tại sao penalty khác?** | Robot tự tìm gait → cần nhiều ràng buộc | Motion reference implicit → ít ràng buộc nhưng cứng hơn |

---

## Kết nối → Doc tiếp theo

Reward sets khác nhau giữa velocity và tracking, nhưng cả 2 đều cần **motion data** cho training.
- Velocity: command (3 số) do environment random sample
- Tracking: motion clip (file NPZ) từ motion capture data

→ **Doc tiếp theo** ([motion_pipeline.md](motion_pipeline.md)) giải thích data đi từ đâu đến đâu: PKL → CSV → NPZ → WandB → Train.

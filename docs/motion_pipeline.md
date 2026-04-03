# Motion Pipeline — Từ Chuyển Động Thật Đến Robot Mô Phỏng

---

## Kết nối với tài liệu trước

> **Doc trước** ([reward_engineering.md](reward_engineering.md)) cho thấy tracking task cần **motion reference** — robot so khớp vị trí, hướng, vận tốc của từng body với clip chuyển động mẫu. **Doc này** giải thích clip chuyển động đó đến từ đâu và qua những bước nào trước khi dùng cho training.

---

## Phần I: Tại Sao Cần Pipeline?

### 1.1 Bài toán

Motion capture thu được chuyển động người thật → cần chuyển sang format mà MuJoCo + RL dùng được.

**Vấn đề**: dữ liệu gốc chỉ có **joint angles** — nhưng tracking reward cần **body positions** trong không gian 3D (reward so khớp vị trí tay, chân, thân). Muốn có body positions → cần chạy **Forward Kinematics (FK)** qua MuJoCo model.

```
Con người: "tay ở đây, chân ở kia"
     ↓ Motion Capture
Raw data: joint angles per frame
     ↓ Forward Kinematics (MuJoCo)
Enriched data: joint angles + body positions + velocities
     ↓ Save + Register
NPZ file on WandB → ready for training
```

### 1.2 Pipeline tổng quan

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│ Motion   │ →  │ Retarget │ →  │    CSV/PKL   │ →  │csv_to_npz│ →  │  WandB   │
│ Capture  │    │ to robot │    │  (raw data)  │    │(MuJoCo FK│    │ Registry │
│          │    │          │    │              │    │ + interp)│    │          │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘    └──────────┘
                                      │                   │
                                 input format         output format
                              per-frame:             NPZ arrays:
                              [x,y,z,qx,qy,qz,qw,   fps, joint_pos,
                               j0,j1,...,jN]          body_pos_w,
                                                      body_quat_w, ...
```

---

## Phần II: Input Format — CSV

### 2.1 Cấu trúc file CSV

Mỗi dòng = 1 frame. Không có header. Dùng dấu phẩy ngăn cách:

```csv
0.00, 0.00, 0.82, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.12, 0.24, -0.12, 0.0, ...
0.02, 0.01, 0.82, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0, -0.15, 0.28, -0.14, 0.0, ...
```

| Cột | Nội dung | Số giá trị |
|-----|----------|------------|
| 0-2 | Base position (x, y, z) | 3 |
| 3-6 | Base orientation quaternion (qx, qy, qz, qw) | 4 |
| 7+ | Joint angles (radians) | N (29 cho G1, 24 cho EngineAI) |

Tổng mỗi dòng: 7 + N giá trị.

> [!WARNING]
> **Quaternion convention quan trọng!**
> - CSV input: `[qx, qy, qz, qw]` (scalar-last, convention phổ biến)
> - MuJoCo internal: `[qw, qx, qy, qz]` (scalar-first)
> - `csv_to_npz` tự chuyển đổi: `motion[:, [3,0,1,2]]` (swap cột 3 về đầu)
>
> Nếu sai convention → robot xoay lộn → training fail ngay bước 1.

---

## Phần III: csv_to_npz — Bên Trong Script Làm Gì?

### 3.1 Tổng quan 4 bước

```
CSV file
  ↓ (1) Load + parse
MotionLoader
  ↓ (2) Interpolate FPS (30Hz → 50Hz)
Interpolated motion
  ↓ (3) Set robot state → MuJoCo FK → record body data
Enriched data
  ↓ (4) Save NPZ + upload WandB
motion.npz + WandB artifact
```

### 3.2 Bước 1: Load CSV

```python
# csv_to_npz.py, MotionLoader._load_motion()

motion = np.loadtxt(input_file, delimiter=",")   # Load toàn bộ CSV

# Tách thành 3 phần:
base_pos  = motion[:, :3]       # (N, 3)  — vị trí xyz
base_rot  = motion[:, 3:7]      # (N, 4)  — quaternion
dof_pos   = motion[:, 7:]       # (N, 29) — joint angles

# Chuyển quaternion: [qx,qy,qz,qw] → [qw,qx,qy,qz] (MuJoCo convention)
base_rot = base_rot[:, [3, 0, 1, 2]]
```

### 3.3 Bước 2: Interpolate FPS

**Tại sao cần interpolate?**

| FPS | Nguồn | Physics dt |
|-----|-------|------------|
| 30 Hz | Motion capture / retarget data | — |
| **50 Hz** | MuJoCo sim (dt = 0.02s) | 0.005s × 4 decimation |

Motion capture thường ở 30 FPS, nhưng MuJoCo env step ở 50 Hz. Cần upsample.

```python
# MotionLoader._interpolate_motion()

# Tạo timestamps ở output FPS
times = [0, 0.02, 0.04, ...]  # mỗi 20ms

# Linear interpolation cho position + joint angles
base_pos_50Hz = lerp(base_pos_30Hz, blend_weights)

# Spherical interpolation (SLERP) cho quaternion
base_rot_50Hz = slerp(base_rot_30Hz, blend_weights)
# Tại sao SLERP? Vì quaternion nằm trên sphere — lerp thường sẽ
# cho kết quả không unit quaternion → sai orientation.
```

**Ví dụ**: 97 frames ở 30Hz (3.2s) → 160 frames ở 50Hz (3.2s). Cùng thời lượng, nhiều frame hơn.

### 3.4 Bước 3: MuJoCo Forward Kinematics

Đây là bước **quan trọng nhất** — lý do tồn tại của script:

```python
# run_sim() — cho mỗi frame:

for frame in motion:
    # Set trạng thái robot theo motion data
    robot.write_root_state_to_sim(base_pos, base_rot, base_vel, base_ang_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # MuJoCo FK: joint angles → body positions
    sim.forward()
    scene.update()
    
    # Record KẾT QUẢ FK:
    log["body_pos_w"].append(robot.data.body_link_pos_w)     # (B, 3) mỗi body
    log["body_quat_w"].append(robot.data.body_link_quat_w)   # (B, 4) mỗi body
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w)
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w)
```

**Tại sao phải dùng MuJoCo FK thay vì tự tính?**

| Phương pháp | Vấn đề |
|-------------|--------|
| Tự tính FK (DH params) | Phải match chính xác MJCF model — dễ sai |
| IsaacLab FK | PhysX body indexing khác MuJoCo |
| **MuJoCo FK** | **Đảm bảo 100% khớp** — cùng model, cùng engine |

> [!IMPORTANT]
> **Đây là lý do csv_to_npz phải chạy qua MuJoCo**: body positions phải khớp CHÍNH XÁC với cách MuJoCo compute chúng trong training. Nếu dùng FK khác engine → body positions hơi sai → tracking reward tính sai → robot học behavior lệch.

### 3.5 Bước 4: Save NPZ + Upload WandB

```python
# Output file: /tmp/motion.npz
np.savez("/tmp/motion.npz",
    fps=[50],                    # Output FPS
    joint_pos=...,               # (T, num_joints)
    joint_vel=...,               # (T, num_joints)
    body_pos_w=...,              # (T, num_bodies, 3)
    body_quat_w=...,             # (T, num_bodies, 4)
    body_lin_vel_w=...,          # (T, num_bodies, 3)
    body_ang_vel_w=...,          # (T, num_bodies, 3)
)

# Upload to WandB
artifact = run.log_artifact("motion.npz", name=output_name, type="motions")
run.link_artifact(artifact, target_path="wandb-registry-motions/...")
```

---

## Phần IV: NPZ Output — Cấu Trúc Chi Tiết

| Array | Shape | Ý nghĩa | Dùng cho |
|-------|-------|---------|---------|
| `fps` | (1,) | Frame rate | Timing |
| `joint_pos` | (T, J) | Joint angles mỗi frame | Action target reference |
| `joint_vel` | (T, J) | Joint velocities | Velocity matching |
| `body_pos_w` | (T, B, 3) | Body positions (world frame) | `motion_body_pos` reward |
| `body_quat_w` | (T, B, 4) | Body orientations (world frame) | `motion_body_ori` reward |
| `body_lin_vel_w` | (T, B, 3) | Body linear velocities | `motion_body_lin_vel` reward |
| `body_ang_vel_w` | (T, B, 3) | Body angular velocities | `motion_body_ang_vel` reward |

Trong đó: T = số frames, J = số joints, B = số bodies tracked.

**Ví dụ G1**: T=160, J=29, B=14 (pelvis + 6 leg bodies + torso + 6 arm bodies).

---

## Phần V: EngineAI PKL → CSV Conversion

### 5.1 PKL Format (giám khảo cung cấp)

```python
# EngineAI_DATA/boxing_30FPS.pkl
data = {
    'fps': 30.0,
    'root_pos': ndarray (97, 3),     # base [x, y, z]
    'root_rot': ndarray (97, 4),     # quaternion [qx, qy, qz, qw]
    'dof_pos':  ndarray (97, 24),    # 24 joint angles (rad)
    'local_body_pos': None,           # csv_to_npz sẽ tính
    'link_body_list': None,
}
```

### 5.2 Script chuyển đổi PKL → CSV

```python
import pickle
import numpy as np

MOTIONS = ['boxing_30FPS', 'dance_30FPS', 'kicking_30FPS']

for name in MOTIONS:
    with open(f'EngineAI_DATA/{name}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    fps = data['fps']
    root_pos = data['root_pos']   # (N, 3)
    root_rot = data['root_rot']   # (N, 4) — [qx, qy, qz, qw]
    dof_pos  = data['dof_pos']    # (N, 24)
    
    # Ghép thành CSV: [x, y, z, qx, qy, qz, qw, j0, j1, ..., j23]
    csv_data = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    np.savetxt(f'EngineAI_DATA/{name}.csv', csv_data, delimiter=',')
    
    print(f'{name}: {root_pos.shape[0]} frames → CSV ({csv_data.shape[1]} values/frame)')
```

> [!WARNING]
> **Quaternion convention phải verify!**
> - PKL có thể dùng `[qx,qy,qz,qw]` (giống CSV expected) → OK
> - Hoặc `[qw,qx,qy,qz]` (MuJoCo native) → cần swap
> - Cách verify: load PKL, check frame 0 quaternion. Nếu `[1,0,0,0]` hoặc `[0,0,0,1]` → biết convention
> - `[1,0,0,0]` = identity trong wxyz (MuJoCo), `[0,0,0,1]` = identity trong xyzw

### 5.3 Sau đó chạy csv_to_npz

```bash
# Cần MODIFY csv_to_npz.py để dùng EngineAI robot config thay vì G1!
# Hoặc viết script riêng. Bước này cần EngineAI asset_zoo entry sẵn.

MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file ~/Documents/Humanoid/EngineAI_DATA/boxing_30FPS.csv \
  --output-name engineai-boxing \
  --input-fps 30 --output-fps 50 --render True
```

> [!IMPORTANT]
> `csv_to_npz.py` hiện tại **hardcode G1 robot config** và **G1 joint names** (29 joints). Để dùng cho EngineAI (24 joints), cần **sửa script** hoặc **tạo version mới** với EngineAI config. Đây là nội dung của Doc tiếp (EngineAI Integration Design).

---

## Phần VI: WandB Registry — Tại Sao?

### 6.1 Workflow

```
csv_to_npz (local) 
  → upload NPZ to WandB as artifact
  → link to "motions" registry
  → training script download NPZ by name

# Train command:
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --registry-name your-org/motions/g1-walk
  
# train.py tự download:
artifact = wandb.Api().artifact("your-org/motions/g1-walk:latest")
motion_file = artifact.download() / "motion.npz"
```

### 6.2 Tại sao không chỉ dùng local file?

| Local file | WandB Registry |
|------------|---------------|
| Chỉ máy bạn có | Team members đều truy cập |
| Không version | Có versioning (:v0, :v1, :latest) |
| Dễ mất khi rm | Persistent cloud storage |
| Không track provenance | Biết ai tạo, khi nào, từ CSV nào |

Tuy nhiên, nếu chỉ dùng local, có thể dùng `--env.commands.motion.motion-file /path/to/motion.npz` thay vì registry.

---

## Phần VII: 3 Motion Clips EngineAI

| Clip | Frames | FPS | Duration | Mô tả |
|------|--------|-----|----------|-------|
| `boxing_30FPS.pkl` | 97 | 30 | 3.2s | Tay đấm boxing |
| `dance_30FPS.pkl` | 252 | 30 | 8.4s | Nhảy múa |
| `kicking_30FPS.pkl` | 152 | 33 | 4.6s | Chân đá |

Sau interpolation (50Hz):

| Clip | Output frames | Duration |
|------|--------------|----------|
| boxing | ~160 | 3.2s |
| dance | ~420 | 8.4s |
| kicking | ~230 | 4.6s |

---

## Kết nối → Doc tiếp theo

Motion pipeline cho EngineAI cần:
1. EngineAI robot trong `asset_zoo/` (MJCF + constants) — chưa có
2. Sửa `csv_to_npz` để dùng EngineAI config — chưa có
3. PKL → CSV conversion script — cần viết

→ **Doc tiếp theo** ([engineai_integration_design.md](engineai_integration_design.md)) thiết kế chi tiết cách tích hợp EngineAI PM_v2 vào mjlab — bao gồm cả motion pipeline adaptation.

# Đề Bài & Phân Tích Chi Tiết

## 1. Đề Bài Gốc

> **Bài test — EngineAI × mjlab**
> 
> **Phần A — Lý thuyết:**
> Giải thích cấu trúc nền code của mjlab, ý nghĩa các module và function, cách nền code hoạt động, và thuật toán training đang được triển khai.
> 
> **Phần B — Thực hành:**
> Train locomotion policy và mimic policy với Unitree G1 và EngineAI PM_v2.
> 
> **Dữ liệu cung cấp:**
> - Repository: [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab)
> - EngineAI PM_v2 motion data: `EngineAI_DATA.zip` chứa 3 motion clips đã retarget (boxing, dance, kicking) ở format PKL
> - Robot model: Clone từ [engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace)

---

## 2. Phân Tích Đề Bài

### 2.1 Tách yêu cầu

Đề bài ngắn gọn nhưng ẩn chứa **8 yêu cầu** ở 2 mức độ:

```
Phần A: LÝ THUYẾT (Hiểu)              Phần B: THỰC HÀNH (Làm)
─────────────────────────────           ─────────────────────────────
A1. Cấu trúc nền code                  B1. G1 locomotion policy
A2. Ý nghĩa module & function          B2. G1 mimic policy
A3. Cách code hoạt động                B3. EngineAI locomotion policy
A4. Thuật toán training                 B4. EngineAI mimic policy (×3 motions)
```

### 2.2 Phân tích mức độ khó

| Yêu cầu | Độ khó | Yêu cầu kỹ năng | Thời gian ước tính |
|----------|--------|------------------|--------------------|
| A1 — Cấu trúc code | ⭐⭐ | Đọc code, kiến trúc SW | ~4h |
| A2 — Module/function | ⭐⭐⭐ | Domain RL + Robotics | ~6h |
| A3 — Data flow | ⭐⭐⭐⭐ | Trace code end-to-end | ~8h |
| A4 — Thuật toán PPO | ⭐⭐⭐ | ML theory + implementation | ~4h |
| B1 — G1 velocity | ⭐ | Setup + chạy train | ~8h (chờ train) |
| B2 — G1 mimic | ⭐⭐ | Motion pipeline | ~8h (chờ train) |
| B3 — EngineAI velocity | ⭐⭐⭐⭐ | MJCF, configs, debug | ~16h |
| B4 — EngineAI mimic | ⭐⭐⭐⭐⭐ | Full pipeline mastery | ~24h |

**Tổng thời gian ước tính**: 70-80 giờ (~6-10 ngày làm việc)

### 2.3 Mục đích đánh giá từng phần

#### Phần A — Giám khảo đánh giá năng lực gì?

| Yêu cầu | Năng lực được đánh giá | Mức "đạt" tối thiểu | Mức "xuất sắc" |
|----------|----------------------|---------------------|----------------|
| **A1** | Software Architecture | Liệt kê đúng modules | Giải thích *tại sao* tổ chức như vậy (Manager-Based pattern, separation of concerns) |
| **A2** | Domain Knowledge | Mô tả function | Giải thích vai trò trong RL loop + kết nối với physical meaning |
| **A3** | System Thinking | Vẽ diagram flow | Trace *cụ thể*: `env.step()` → ActionManager → PD controller → MuJoCo Warp → ObsManager → ... |
| **A4** | ML Theory + Practice | Mô tả PPO generic | PPO *trong ngữ cảnh mjlab*: tại sao hyperparams cụ thể, adaptive LR, GAE, RSL-RL implementation |

#### Phần B — Giám khảo phân biệt ứng viên thế nào?

| Mức | G1 | EngineAI | Đánh giá |
|-----|-----|----------|----------|
| **Yếu** | Không chạy được | — | Không hiểu setup |
| **Đạt** | Train xong velocity | — | Biết dùng framework |
| **Khá** | Train xong velocity + mimic | Train velocity | Hiểu motion pipeline |
| **Giỏi** | Đầy đủ | Train velocity + 1-2 mimic | Hiểu framework đủ sâu để extend |
| **Xuất sắc** | Đầy đủ + phân tích | Train đầy đủ 4 motions + so sánh | End-to-end mastery |

> [!IMPORTANT]
> **EngineAI là phần quyết định.** G1 đã có config sẵn → ai cũng chạy được. EngineAI buộc ứng viên phải *hiểu* framework đủ sâu để *tự extend*: tạo MJCF, viết constants, adapt reward/obs, convert motion data, và debug khi robot ngã.

### 2.4 Thách thức ẩn trong đề bài

#### 1. Motion data ở format PKL — mjlab cần NPZ
- Giám khảo cung cấp PKL, nhưng mjlab chỉ hỗ trợ CSV → NPZ
- Cần viết script chuyển đổi PKL → CSV, hiểu quaternion convention
- **Bẫy tiềm ẩn**: Thứ tự quaternion (MuJoCo: `[w,x,y,z]` vs đề cung cấp: `[x,y,z,w]`)

#### 2. Robot model phải tự tích hợp
- G1 có sẵn trong mjlab, EngineAI thì không
- Cần hiểu MJCF format, joint ordering, actuator config
- PD gains cần tune — giá trị sai → robot rung/ngã

#### 3. GPU constraint
- Laptop chỉ có RTX 3050 Ti (4GB VRAM) → chỉ chạy 256-512 environments
- Training time tăng 5-10× so với GPU mạnh
- **Giải pháp**: Dùng server RTX 3090 + Google Colab A100 để train parallel

#### 4. Render video headless
- `play.py` khi không có display sẽ mở Viser viewer (web server) → treo
- Cần viết custom headless renderer hoặc dùng `MUJOCO_GL=egl`

---

## 3. Chiến Lược Thực Hiện

### 3.1 Approach: Documentation-First

```
Giai đoạn 1: ĐỌC & HIỂU (2-3 ngày)
    Đọc code → Viết tài liệu → Hiểu sâu architecture

Giai đoạn 2: G1 BASELINE (1-2 ngày)
    Setup → Train G1 velocity → Evaluate → Chứng minh hiểu framework

Giai đoạn 3: ENGINEAI INTEGRATION (3-5 ngày)
    Tạo asset → Tạo config → Debug → Train velocity
    → Convert motion → Train mimic (×3)

Giai đoạn 4: BÁO CÁO (0.5 ngày)
    So sánh → Tổng hợp → Đóng gói nộp
```

### 3.2 Tối ưu thời gian training

| Platform | GPU | VRAM | Envs | Dùng cho |
|----------|-----|------|------|----------|
| Laptop | RTX 3050 Ti | 4GB | 512 | G1 velocity (baseline) |
| Server | RTX 3090 | 24GB | 4,096 | EngineAI velocity + boxing + kicking |
| Colab | A100 | 80GB | 16,384 | EngineAI dance (lớn nhất) |

**Parallel training**: Boxing/kicking chạy song song trên server, dance chạy trên Colab → tiết kiệm 50% thời gian.

---

## 4. Checklist Đánh Giá Hoàn Thành

### Phần A — Tài liệu

- [x] **A1.** Cấu trúc code: `codebase_analysis.md` (579 dòng)
- [x] **A2.** Module/function: `big_picture_overview.md` (600 dòng) + `codebase_analysis.md`
- [x] **A3.** Data flow: `big_picture_overview.md` Chương 3 (trace env.step chi tiết)
- [x] **A4.** Thuật toán: `ppo_training_deep_dive.md` (512 dòng) + `reward_engineering.md` (398 dòng)

### Phần B — Thực hành

- [x] **B1.** G1 velocity: 30K iters, reward=34.16, fell=0% ✅
- [ ] ~~**B2.** G1 mimic~~ (Skip — focus vào EngineAI, có LAFAN1 dataset nhưng ưu tiên thời gian)
- [x] **B3.** EngineAI velocity: 20K iters, reward=65.98, fell=4% ✅
- [x] **B4.** EngineAI boxing: 30K iters ✅
- [x] **B4.** EngineAI kicking: 20K iters ✅
- [x] **B4.** EngineAI dance: 5K iters (A100), reward=38.27 ✅

### Deliverables

- [x] Source code (asset_zoo + task configs)
- [x] 4 checkpoints (.pt) + 1 ONNX
- [x] 4 demo videos (1080p)
- [x] 3 motion NPZ files
- [x] Training scripts (server + Colab)
- [x] Tài liệu phân tích (10 docs)
- [x] So sánh G1 vs EngineAI
- [x] Báo cáo tổng hợp
- [x] README hướng dẫn giám khảo

---

## 5. Kết Luận

Đề bài tuy ngắn nhưng đánh giá toàn diện: từ khả năng đọc hiểu code (Part A), đến kỹ năng thực hành (Train G1), đến năng lực tích hợp và giải quyết vấn đề (EngineAI). Phần EngineAI mimic (B4) là phần khó nhất và có giá trị phân biệt cao nhất — yêu cầu ứng viên thành thạo toàn bộ pipeline từ robot model → motion data → RL config → training → evaluation.

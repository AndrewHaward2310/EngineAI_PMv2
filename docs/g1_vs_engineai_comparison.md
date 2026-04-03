# So Sánh G1 vs EngineAI PM_v2 — Kỹ Thuật & Kết Quả Training

## 1. So Sánh Hardware Robot

| Thông số | Unitree G1 | EngineAI PM_v2 | Nhận xét |
|----------|-----------|----------------|----------|
| **Chiều cao** | ~1.27m | ~0.82m | PM_v2 thấp hơn 35% → trọng tâm thấp hơn |
| **Tổng DOFs** | 29 | 24 | G1 nhiều DOF hơn ở cổ tay |
| **Chân (mỗi bên)** | 6 DOF | 6 DOF | **Giống nhau**: hip×3 + knee + ankle×2 |
| **Lưng/eo** | 3 DOF (yaw+roll+pitch) | 1 DOF (yaw only) | G1 linh hoạt hơn xoay thân |
| **Tay (mỗi bên)** | 7 DOF | 5 DOF | G1 có wrist (3 DOF), PM_v2 không |
| **Đầu** | 0 DOF | 1 DOF (yaw) | PM_v2 có head control |
| **Leg torque** | ~60-120 Nm | ~61-164 Nm | PM_v2 motors mạnh hơn |
| **Physics engine** | MuJoCo Warp (GPU) | MuJoCo Warp (GPU) | Cùng simulator |

### Ảnh hưởng đến RL Training

| Khác biệt | Ảnh hưởng |
|-----------|-----------|
| Ít DOFs (24 vs 29) | Action space nhỏ hơn → policy network nhỏ hơn (24 vs 29 output) → converge nhanh hơn |
| Thấp hơn | Trọng tâm thấp → moment quán tính nhỏ → **dễ balance hơn** |
| Waist 1 DOF | Hạn chế khả năng xoay thân khi đi sideway → gait có thể ít tự nhiên |
| Không có wrist | Mimic chỉ track đến elbow → motion reference đơn giản hơn |
| Motor mạnh hơn | Cần PD gains khác biệt → phải tune kp/kd cho phù hợp |

---

## 2. So Sánh Configuration

### 2.1 Observation Space

| Feature | G1 (99D actor) | PM_v2 (135D actor) | Giải thích |
|---------|---------------|-------------------|-----------|
| Joint pos | 29 | 24 | Bằng số DOF |
| Joint vel | 29 | 24 | |
| Base ang vel | 3 | 3 | |
| Projected gravity | 3 | — | PM_v2 tracking không dùng |
| Velocity command | 3 (vx,vy,ωz) | — | Chỉ cho velocity task |
| Motion command | — | 48 | Cho tracking task |
| Previous action | 29 | 24 | |
| **Total** | **99** (actor) | **135** (actor) | PM_v2 lớn hơn vì motion command 48D |

### 2.2 Reward Design

#### Velocity Task

| Reward Term | G1 Weight | PM_v2 Weight | Giữ/Đổi |
|------------|-----------|-------------|---------|
| Linear velocity tracking | 2.0 | 2.0 | Giữ |
| Angular velocity tracking | 2.0 | 2.0 | Giữ |
| Upright | 1.0 | 1.0 | Giữ |
| Pose | 1.0 | 1.0 | Giữ |
| Action rate L2 | -0.1 | -0.1 | Giữ |
| Energy | -0.001 | -0.001 | Giữ |
| DOF pos limits | -1.0 | -1.0 | Giữ |
| Self collisions | -1.0 | -1.0 | Giữ |
| **base_height target** | **0.75m** | **0.79m** | **Đổi** — PM_v2 thấp hơn |
| **termination height** | **0.3m** | **0.3m** | Giữ (tỉ lệ ok) |
| Air time | 2.0 | **0.0** | **Đổi** — bỏ vì gây lỗi cho PM_v2 |

> [!NOTE]
> `air_time = 2.0` từ G1 config gây vấn đề cho PM_v2: robot "nhảy" cả 2 chân để maximize air time thay vì đi. Đặt `0.0` → robot đi bình thường.

#### Tracking (Mimic) Task

| Reward Term | Weight | Mô tả |
|------------|--------|--------|
| motion_global_root_pos | 0.5 | Track vị trí root (x,y,z) |
| motion_global_root_ori | 0.5 | Track hướng root |
| motion_body_pos | 1.0 | Track vị trí 14 bodies |
| motion_body_ori | 1.0 | Track hướng 14 bodies |
| motion_body_lin_vel | 1.0 | Track vận tốc tịnh tiến |
| motion_body_ang_vel | 1.0 | Track vận tốc góc |
| action_rate_l2 | -0.05 | Smoothness penalty |
| joint_limit | -10.0 | Joint limit violation |
| self_collisions | -10.0 | Self-collision penalty |

### 2.3 PD Gains

| Nhóm khớp | G1 kp | PM_v2 kp | G1 kd | PM_v2 kd |
|-----------|------|---------|------|---------|
| Hip | 100–150 | 100 | 3–5 | 2 |
| Knee | 100–150 | 100 | 3–5 | 2 |
| Ankle | 50–100 | 40 | 1–3 | 1 |
| Waist | 80–120 | 80 | 2–4 | 2 |
| Shoulder | 50–80 | 60 | 1–3 | 1.5 |
| Elbow | 30–50 | 40 | 0.5–1 | 1 |
| Head | — | 20 | — | 0.5 |

> PM_v2 gains thấp hơn vì motor response khác biệt. Quá trình tune: bắt đầu từ G1 gains → giảm dần kd cho đến khi robot không rung.

---

## 3. So Sánh Kết Quả Training

### 3.1 Velocity (Locomotion)

| Metric | G1 (RTX 3050 Ti) | PM_v2 (RTX 3090) |
|--------|-------------------|-------------------|
| **Iterations** | 30,000 | 20,000 |
| **Num envs** | 512 | 4,096 |
| **Training time** | 7h 20m | ~7h |
| **Final reward** | 34.16 | **65.98** |
| **Episode length** | 987/1000 | 1000/1000 |
| **Fall rate** | 0% | **4%** |
| **Steps/sec** | ~14,500 | ~80,000 |

**Phân tích:**
- PM_v2 reward cao hơn gần 2× → gait quality tốt hơn (hoặc reward scale khác)
- PM_v2 vẫn ngã 4% → có thể cải thiện bằng thêm iterations hoặc tune termination
- RTX 3090 nhanh hơn ~5.5× so với 3050 Ti (80K vs 14.5K steps/s)

### 3.2 Mimic (Motion Tracking)

| Metric | Boxing (3090) | Kicking (3090) | Dance (A100) |
|--------|--------------|----------------|--------------|
| **Iterations** | 30,000 | 20,000 | 5,000 |
| **Num envs** | 4,096 | 4,096 | 16,384 |
| **Training time** | ~14h | ~7h | ~4.3h |
| **Final reward** | ~35 | ~32 | **38.27** |
| **Motion duration** | 3.2s | 4.6s | 8.4s |
| **Motion frames** | 97 | 152 | 252 |

**Phân tích:**
- **Dance** đạt reward cao nhất (38.27) dù ít iterations nhất → A100 với 16K envs hiệu quả hơn
- **Kicking** thấp nhất (32) → motion khó hơn (1 chân, cần balance)
- **Boxing** trung bình (35) → chủ yếu upper body motion

### 3.3 Tracking Quality (Dance model — chi tiết)

| Tracking Metric | Value | Đánh giá |
|----------------|-------|----------|
| motion_body_pos | 0.9613 | ✅ Rất tốt — track vị trí body chính xác |
| motion_body_ori | 0.8292 | ✅ Tốt — track hướng body |
| motion_body_lin_vel | 0.8603 | ✅ Tốt — track vận tốc |
| motion_body_ang_vel | 0.7980 | ⚠️ Khá — angular vel khó track nhất |
| motion_global_root_pos | 0.3931 | ⚠️ Trung bình — root drift theo thời gian |
| motion_global_root_ori | 0.4723 | ⚠️ Trung bình — heading có sai lệch |
| action_rate_l2 | -0.5190 | Action khá smooth |
| joint_limit | -0.0038 | ✅ Rất ít vi phạm joint limits |
| self_collisions | -0.0082 | ✅ Hầu như không có self-collision |

### 3.4 Motion Error Analysis (Dance)

| Error Type | Value | Ý nghĩa |
|-----------|-------|---------|
| error_anchor_pos | 0.1413 m | Sai lệch vị trí ~14cm |
| error_anchor_rot | 0.0736 rad | Sai lệch góc ~4.2° |
| error_anchor_lin_vel | 0.2371 m/s | Sai lệch vận tốc |
| error_body_lin_vel | 0.2833 m/s | Body velocity tracking |
| error_body_ang_vel | 1.1533 rad/s | Angular velocity — khó nhất |

---

## 4. So Sánh Qualitative (Video)

### Velocity (Locomotion)

| Tiêu chí | G1 | PM_v2 |
|----------|-----|-------|
| Đi thẳng | ✅ Mượt | ✅ Mượt |
| Đi ngang | ✅ OK | ⚠️ Hơi kém (waist 1 DOF) |
| Xoay | ✅ Tốt | ✅ Tốt |
| Ổn định 60s | ✅ Không ngã | ✅ Không ngã (96%) |
| Tự nhiên | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Mimic (Motion Tracking)

| Video | Chất lượng | Nhận xét |
|-------|-----------|----------|
| Boxing 60s | ⭐⭐⭐⭐ | Track đấm tốt, thân ổn định |
| Kicking 60s | ⭐⭐⭐ | Đá được nhưng hơi mất balance |
| Dance 60s | ⭐⭐⭐⭐ | Bắt chước nhảy khá tốt, body pos tracking cao (0.96) |

---

## 5. Bài Học & Kinh Nghiệm

### 5.1 Debug Journey

| Vấn đề gặp | Nguyên nhân | Giải pháp |
|------------|-------------|-----------|
| Robot rung khi đứng | kd quá cao | Giảm kd từ 5 → 2 |
| Robot nhảy 2 chân | air_time reward=2.0 | Đặt air_time=0.0 |
| Robot ngã ngay | PD gains sai tần số | Chuyển từ 5Hz → 10Hz |
| Training chậm trên 3050 Ti | 4GB VRAM | Dùng server 3090 + Colab A100 |
| Render treo trên Colab | Viser viewer blocking | Viết headless renderer |
| Checkpoint sort sai | Path alphabetical vs numeric | Sort theo iteration number |

### 5.2 GPU Scaling Analysis

| Platform | GPU | Envs | Steps/s | Time/20K iters |
|----------|-----|------|---------|---------------|
| Laptop | RTX 3050 Ti | 512 | 14,500 | ~15h |
| Server | RTX 3090 | 4,096 | 80,000 | ~7h |
| Colab | A100 80GB | 16,384 | 81,155 | ~4.3h (5K iters) |

> [!TIP]
> Tăng `num_envs` tỷ lệ thuận VRAM nhưng **không** tỉ lệ thuận tốc độ. Với PPO on-policy, bottleneck chuyển từ GPU compute sang PPO update khi envs quá lớn. Sweet spot: 4K-8K envs cho 24GB, 8K-16K cho 80GB.

---

## 6. Kết Luận

1. **EngineAI PM_v2 tích hợp thành công** vào mjlab framework — 4 policies trained (velocity + 3 mimic motions)
2. **PM_v2 dễ balance hơn G1** nhờ trọng tâm thấp (0.82m vs 1.27m) — reward velocity cao hơn (65.98 vs 34.16)
3. **Motion tracking tốt nhất ở body position** (0.96) — yếu nhất ở angular velocity (1.15 rad/s error)
4. **Multi-GPU strategy hiệu quả**: song song RTX 3090 + A100 giảm 50% tổng thời gian training
5. **Key insight**: Với humanoid RL, debug config (PD gains, reward weights, termination thresholds) chiếm nhiều thời gian hơn viết code

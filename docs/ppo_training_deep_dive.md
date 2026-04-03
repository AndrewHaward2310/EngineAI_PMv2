# PPO Training Deep-Dive — Từ Toán Học Đến Code Thực Tế

---

## Kết nối với tài liệu trước

> **Doc trước** ([codebase_analysis.md](codebase_analysis.md)) giải thích `env.step()` tạo ra `(obs, reward, done)` — robot tương tác với thế giới mô phỏng. **Doc này** giải thích PPO **dùng** những output đó như thế nào để cải thiện policy — biến robot từ "ngã liên tục" thành "đi vững vàng".

---

## Phần I: Xuất Phát Điểm — Tại Sao Cần Policy Gradient?

### 1.1 Bài toán tối ưu

Robot cần tìm **policy** π(a|s) — hàm nhận trạng thái s, output xác suất chọn hành động a — sao cho **tổng reward kỳ vọng** lớn nhất:

```
Mục tiêu: max_θ  J(θ) = E[Σ γ^t × r_t]

Trong đó:
  θ     = tham số neural network (actor)
  γ     = 0.99 (discount factor — bước gần quan trọng hơn bước xa)
  r_t   = reward tại timestep t
```

**Ví dụ trực quan**: Robot thử 512 cách đi song song. Mỗi cách cho reward khác nhau. Cần tìm gradient ∇J(θ) để update θ → robot đi tốt hơn lần sau.

### 1.2 Policy Gradient cơ bản

Ý tưởng: **tăng xác suất** của hành động tốt, **giảm xác suất** hành động xấu:

```
∇J(θ) = E[ ∇log π(a|s) × R ]

Dịch: gradient = (hướng tăng xác suất hành động a) × (hành động a tốt cỡ nào)
```

**Vấn đề của Policy Gradient thuần**:

| Vấn đề | Hệ quả |
|--------|--------|
| **Variance cao** | Gradient "nhảy múa" → training không ổn định |
| **Sample-inefficient** | Mỗi batch data chỉ dùng 1 lần rồi bỏ |
| **Step size nhạy** | Learning rate lớn → policy collapse; nhỏ → learn chậm |

→ Cần algorithm thông minh hơn: **PPO**.

---

## Phần II: PPO — Proximal Policy Optimization

### 2.1 Ý tưởng cốt lõi

**"Cập nhật policy nhưng không được thay đổi quá nhiều"**

Giả sử policy hiện tại π_old đang tạm ổn (robot biết đi chậm). Nếu update quá mạnh → policy mới π_new có thể khác hoàn toàn → robot quên hết → ngã liên tục → phải learn lại từ đầu.

PPO giải quyết bằng **clipping**: ép buộc π_new không được khác π_old quá 20%.

### 2.2 Clipped Surrogate Objective — Trái tim của PPO

```python
# Tỉ lệ xác suất mới/cũ
r(θ) = π_new(a|s) / π_old(a|s)

# Loss function
L_PPO = -E[ min(
    r(θ) × A,                              # Surrogate loss thường
    clip(r(θ), 1-ε, 1+ε) × A              # Surrogate loss bị clip
) ]

# Trong đó:
#   A = advantage (hành động này tốt hơn trung bình bao nhiêu)
#   ε = 0.2 (clip range)
```

**Giải thích bằng ví dụ số**:

Giả sử advantage A = +5 (hành động tốt hơn trung bình nhiều):

| r(θ) | Không clip | Clip [0.8, 1.2] | PPO chọn |
|------|-----------|-----------------|----------|
| 0.7 | 0.7 × 5 = 3.5 | 0.8 × 5 = 4.0 | **max**(3.5, 4.0) = 4.0 |
| 1.0 | 1.0 × 5 = 5.0 | 1.0 × 5 = 5.0 | 5.0 |
| 1.3 | 1.3 × 5 = 6.5 | 1.2 × 5 = 6.0 | **max**(6.5, 6.0) = 6.5 |
| 1.5 | 1.5 × 5 = 7.5 | 1.2 × 5 = 6.0 | **max**(7.5, 6.0) = 7.5 |

> [!NOTE]
> Hàm loss dùng `max()` chứ không phải `min()` vì ta **minimize** loss (loss = negative reward). `max(surrogate, clipped)` = chọn loss LỚN hơn = **pessimistic** = an toàn hơn.

Khi A > 0 (hành động tốt): clip ngăn r(θ) vượt quá 1+ε → không tham lam quá.
Khi A < 0 (hành động xấu): clip ngăn r(θ) giảm dưới 1-ε → không trừng phạt quá mạnh.

→ Policy thay đổi **từ từ**, ổn định.

### 2.3 Trong code RSL-RL (ppo.py)

```python
# File: rsl_rl/algorithms/ppo.py, method update()

# Tính ratio
ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob)

# Surrogate loss (không clip)
surrogate = -batch.advantages * ratio

# Surrogate loss (có clip)
surrogate_clipped = -batch.advantages * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)

# PPO loss = max (pessimistic choice)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

### 2.4 Tại sao PPO mà không phải algorithm khác?

| Algorithm | Loại | Ưu | Nhược | Dùng cho |
|-----------|------|-----|-------|----------|
| REINFORCE | On-policy | Đơn giản | Variance cực cao | Toy problems |
| TRPO | On-policy | Lý thuyết đẹp | Tính toán nặng (Hessian) | Research |
| **PPO** | **On-policy** | **Đơn giản + ổn định** | **Sample-inefficient** | **Locomotion** ✅ |
| SAC | Off-policy | Sample-efficient | Kém cho locomotion | Manipulation |
| TD3 | Off-policy | Ổn định hơn SAC | Cần tuning nhiều | Manipulation |

**Tại sao on-policy cho locomotion?**
- Locomotion cần **ổn định** (robot không được ngã) → on-policy an toàn hơn
- GPU simulation cho phép thu thập data cực nhanh (512 envs × 14K steps/s) → sample-inefficiency không còn là vấn đề
- Cộng đồng robotics (ETH Zurich, UC Berkeley, Google) đã validate PPO rộng rãi cho legged robots

---

## Phần III: GAE — Generalized Advantage Estimation

### 3.1 Tại sao cần Advantage thay vì raw reward?

**Vấn đề**: reward = 10 có tốt không?
- Nếu trung bình 5 → 10 là rất tốt (A = +5)
- Nếu trung bình 15 → 10 là tệ (A = -5)

**Advantage = reward - baseline** = "hành động này tốt hơn kỳ vọng bao nhiêu?"

```
A(s, a) = Q(s, a) - V(s)

Q(s, a) = expected return khi ở s, làm a
V(s)    = expected return khi ở s (trung bình mọi a) ← Critic network ước tính
```

### 3.2 Cách tính GAE — Bias-Variance Trade-off

```python
# File: rsl_rl/algorithms/ppo.py, method compute_returns()

# TD error tại step t:
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)

# GAE: weighted sum of multi-step TD errors
A_t = δ_t + (γλ) × δ_{t+1} + (γλ)² × δ_{t+2} + ...
    = Σ_{k=0}^{∞} (γλ)^k × δ_{t+k}
```

**Lambda (λ) controls bias-variance**:

| λ | Advantage | Bias | Variance | Ý nghĩa |
|---|-----------|------|----------|---------|
| 0 | A = δ_t = r_t + γV(s') - V(s) | Cao (phụ thuộc V accuracy) | Thấp | Chỉ nhìn 1 step |
| 1 | A = Σ r_t - V(s) | Thấp | Cao | Monte Carlo (nhìn toàn bộ episode) |
| **0.95** | **Balance** | **Vừa phải** | **Vừa phải** | **"Nhìn" ~20 steps** ✅ |

**Trực quan**: λ = 0.95 nghĩa là mỗi δ cách xa hơn bị discount (0.95)^k. Sau ~20 steps, weight gần 0 → advantage chỉ "nhìn" ~20 steps tương lai.

### 3.3 Trong code

```python
# rsl_rl/algorithms/ppo.py, compute_returns()

advantage = 0
for step in reversed(range(num_steps)):
    next_values = last_values if step == num_steps - 1 else st.values[step + 1]
    next_is_not_terminal = 1.0 - st.dones[step].float()
    
    # TD error
    delta = st.rewards[step] + next_is_not_terminal * gamma * next_values - st.values[step]
    
    # GAE recursive formula (tính ngược từ cuối)
    advantage = delta + next_is_not_terminal * gamma * lam * advantage
    
    # Return = Advantage + Value
    st.returns[step] = advantage + st.values[step]

# Normalize advantages → mean=0, std=1
st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
```

> [!IMPORTANT]
> **Bootstrapping on timeout**: Khi episode kết thúc do hết thời gian (truncated, không phải ngã), reward được **cộng thêm** γ × V(s_last). Tại sao? Vì robot không thực sự "thua" — chỉ hết quota. Nếu không bootstrap, robot sẽ "sợ" cuối episode và hành xử lạ.

```python
# ppo.py, process_env_step()
if "time_outs" in extras:
    self.transition.rewards += self.gamma * self.transition.values * extras["time_outs"]
```

---

## Phần IV: Training Loop — Trace Code Đầy Đủ

### 4.1 Tổng quan vòng lặp

```
Lệnh: uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 512

train.py
  ├─ TaskRegistry → load (env_cfg, rl_cfg)
  ├─ ManagerBasedRlEnv(env_cfg)      ← 512 robots trên GPU
  ├─ RslRlVecEnvWrapper(env)         ← bridge mjlab ↔ RSL-RL
  ├─ MjlabOnPolicyRunner(wrapper)    ← init Actor, Critic, PPO
  └─ runner.learn(30000 iterations)
       └─ For iter = 1 → 30000:
            ├─ ROLLOUT (24 steps)          ← thu thập kinh nghiệm
            ├─ COMPUTE RETURNS (GAE)       ← tính advantage
            ├─ PPO UPDATE (5 epochs × 4mb) ← cập nhật networks
            └─ LOG + SAVE                  ← ghi metrics
```

### 4.2 Phase 1: Rollout — Thu thập kinh nghiệm

```
For step = 1 → 24:
  obs = wrapper.get_observations()          # GPU tensor [512, 99]
  action = actor(obs, stochastic=True)      # Neural net → sample action [512, 29]
  obs', reward, done, info = wrapper.step(action)  # Simulation + RL signals
  storage.add(obs, action, reward, value, done, log_prob)
```

**Tại sao 24 steps?**

```
24 steps × 0.02s (env step) = 0.48s kinh nghiệm thực

Robot ngã trung bình sau ~0.3-0.5s khi untrained
→ 24 steps vừa đủ để "thấy" hệ quả: đi tốt → sống sót, đi xấu → ngã
→ PPO có đủ tín hiệu để phân biệt hành động tốt/xấu
```

Nếu quá ngắn (5 steps): robot chưa kịp ngã → không biết hành động đó tệ.
Nếu quá dài (100 steps): tốn memory, rollout chậm, credit assignment khó.

### 4.3 Phase 2: Compute Returns

Sau 24 steps, PPO tính:
- **GAE advantages** cho mọi (state, action) pair
- **Returns** = advantages + values (target cho critic)
- **Normalize** advantages (mean=0, std=1) → ổn định training

### 4.4 Phase 3: PPO Update

```python
# Mini-batch training
For epoch = 1 → 5:                    # Reuse data 5 lần
  For mini_batch in shuffle(storage, 4 parts):  # Chia data thành 4 phần
    
    # 1. Forward pass với policy MỚI
    new_log_prob = actor(batch.obs)
    new_value = critic(batch.obs)
    
    # 2. Tính ratio = π_new / π_old
    ratio = exp(new_log_prob - old_log_prob)
    
    # 3. Clipped surrogate loss
    surr1 = -advantage * ratio
    surr2 = -advantage * clip(ratio, 0.8, 1.2)
    policy_loss = max(surr1, surr2).mean()
    
    # 4. Value loss (cũng có clip!)
    value_loss = clipped_mse(new_value, returns)
    
    # 5. Entropy bonus (khuyến khích exploration)
    entropy_loss = -entropy.mean()
    
    # 6. Tổng loss
    loss = policy_loss + 1.0 * value_loss + 0.01 * entropy_loss
    #                    ↑ value_loss_coef   ↑ entropy_coef
    
    # 7. Adaptive LR
    kl = KL(π_old, π_new)
    if kl > 2 × 0.01:  lr /= 1.5   # Policy đổi quá nhiều → chậm lại
    if kl < 0.01 / 2:  lr *= 1.5   # Policy đổi quá ít → nhanh hơn
    
    # 8. Gradient update
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(params, max_norm=1.0)  # Chống gradient explosion
    optimizer.step()
```

**Tại sao 5 epochs, 4 mini-batches?**

```
Data mỗi rollout:  512 envs × 24 steps = 12,288 samples
Mini-batch size:   12,288 / 4 = 3,072 samples
Tổng updates:      5 × 4 = 20 gradient steps per iteration

Nếu epochs quá nhiều (20): → overfitting vào data cũ → PPO hỏng (vì data on-policy)
Nếu epochs quá ít (1):     → lãng phí data, learn chậm
5 epochs = balance đã được validate bởi ETH Zurich cho locomotion
```

---

## Phần V: Adaptive Learning Rate — Cơ Chế Tự Điều Chỉnh

### 5.1 Tại sao cần adaptive?

Fixed LR có vấn đề:
- **Đầu training** (policy random): cần LR lớn để explore nhanh
- **Giữa training** (policy ổn): cần LR vừa để fine-tune
- **Cuối training** (policy tốt): cần LR nhỏ để không phá

### 5.2 Cơ chế KL-based

```python
# Sau mỗi PPO update, tính KL divergence giữa π_old và π_new
kl = mean(KL(π_old || π_new))

desired_kl = 0.01  # Mức thay đổi "vừa phải"

if kl > 2 × desired_kl:     # Policy thay đổi QUÁ NHIỀU
    lr = max(1e-5, lr / 1.5)   # → Giảm lr → safe
elif kl < desired_kl / 2:   # Policy thay đổi QUÁ ÍT  
    lr = min(1e-2, lr * 1.5)   # → Tăng lr → learn nhanh hơn
# else: giữ nguyên lr
```

**KL divergence** = đo mức khác biệt giữa 2 phân phối. KL = 0 nghĩa là π_new = π_old (không học gì). KL lớn nghĩa là policy thay đổi mạnh.

### 5.3 Minh họa bằng data G1 thực

Từ kết quả training G1:

```
Iter    1 → lr = 1e-3 (khởi đầu)
Iter  500 → lr giảm dần (policy đang learn nhanh, KL cao)
Iter 5000 → lr ≈ 4e-4 (ổn định)
Iter 29999 → lr ≈ 1e-4 (converge, fine-tuning)
```

Biểu đồ reward:
```
Reward
  35 |                    ****  **    ****  ****
  30 |                 ***    **  ****    **
  25 |              ***
  20 |           ***
  15 |        ***
  10 |      **
   5 |    **
   0 | ***
  -5 |*
     +──────────────────────────────────────── Iteration
     0    3K   6K   9K   12K  15K  18K  21K  24K  27K  30K
```

> [!NOTE]
> **Reward dip ở iter 12K-15K**: Đây là do **curriculum expansion** — command velocity range mở rộng từ [-1, 1] lên [-2, 3] m/s. Robot phải học lại cho range mới → reward tạm giảm → rồi hồi phục.

---

## Phần VI: 3 Losses — Mỗi Cái Có Vai Trò Riêng

### 6.1 Total Loss

```
L_total = L_surrogate + c₁ × L_value - c₂ × L_entropy
```

| Loss | Vai trò | Coefficient |
|------|---------|-------------|
| **Surrogate** (policy) | Cải thiện actor → chọn action tốt hơn | 1.0 (implicit) |
| **Value** (critic) | Cải thiện critic → ước tính V(s) chính xác hơn | c₁ = 1.0 |
| **Entropy** (exploration) | Khuyến khích actor thử hành động đa dạng | c₂ = 0.01 |

### 6.2 Entropy Bonus — Tại sao quan trọng?

```
Không có entropy: actor nhanh chóng "quyết đoán" → chỉ thử 1 cách đi → local optimum
                  Ví dụ: robot tìm ra cách đứng yên (reward = 0, nhưng không ngã)
                  → không bao giờ khám phá "đi" vì đi có risk ngã

Có entropy:       actor bị "ép" thử nhiều action → khám phá nhiều cách → tìm ra cách đi tốt
                  entropy_coef = 0.01: nhẹ nhàng, không ép quá mạnh
```

### 6.3 Clipped Value Loss — Bảo vệ critic

```python
# Thường: value_loss = (V_new(s) - R_target)²

# PPO clip cả value:
V_clipped = V_old + clip(V_new - V_old, -ε, +ε)
value_loss = max(
    (V_new - R_target)²,
    (V_clipped - R_target)²
).mean()
```

Tại sao? Vì critic cũng không nên thay đổi quá nhanh → ổn định baseline cho advantage estimation.

---

## Phần VII: Observation Normalization — Chi Tiết Quan Trọng

### 7.1 Vấn đề input scale

Observations của robot có range rất khác nhau:

| Observation | Range | Scale |
|-------------|-------|-------|
| Joint position | [-3.14, 3.14] rad | ~6 |
| Joint velocity | [-20, 20] rad/s | ~40 |
| Base angular vel | [-5, 5] rad/s | ~10 |
| Gravity projection | [-1, 1] | ~2 |

Neural network hoạt động tốt nhất khi input **chuẩn hóa** (mean≈0, std≈1).

### 7.2 Running normalization (EmpiricalNormalization)

```python
# Actor và Critic đều có obs_normalizer
# Cập nhật running statistics:
obs_normalizer.update(new_obs)
  → running_mean = 0.99 * old_mean + 0.01 * batch_mean
  → running_var  = 0.99 * old_var  + 0.01 * batch_var

# Normalize:
obs_normalized = (obs - running_mean) / sqrt(running_var + 1e-8)
```

> [!IMPORTANT]
> `obs_normalization=True` được bật trong config G1. Đây là chi tiết **rất quan trọng**: nếu tắt, training có thể diverge vì obs range quá khác nhau.

---

## Phần VIII: Tổng Kết — Data Flow Hoàn Chỉnh 1 Iteration

```
┌─────────────────────────────────────────────────────────────────┐
│                    1 Training Iteration                         │
│                                                                 │
│  ROLLOUT (24 steps × 512 envs = 12,288 transitions)            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ For step = 1→24:                                     │       │
│  │   obs ──→ Actor ──→ action ──→ env.step()            │       │
│  │              │                    │                   │       │
│  │              └── Critic ── value  │                   │       │
│  │                                   │                   │       │
│  │   ←── obs', reward, done ─────────┘                   │       │
│  │   store(obs, action, reward, value, done, log_prob)   │       │
│  └──────────────────────────────────────────────────────┘       │
│                          ↓                                      │
│  COMPUTE RETURNS                                                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ reverse scan: δ_t = r_t + γV(s') - V(s)             │       │
│  │               A_t = δ_t + γλ × A_{t+1}              │       │
│  │ normalize: A = (A - mean) / std                       │       │
│  └──────────────────────────────────────────────────────┘       │
│                          ↓                                      │
│  PPO UPDATE (5 epochs × 4 mini-batches = 20 updates)            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ For each mini-batch:                                  │       │
│  │   ratio = π_new / π_old                               │       │
│  │   loss = surrogate + value_loss - entropy             │       │
│  │   adaptive_lr(KL)                                     │       │
│  │   optimizer.step()                                    │       │
│  └──────────────────────────────────────────────────────┘       │
│                          ↓                                      │
│  LOG: reward, episode_len, lr, losses → WandB                   │
│  SAVE: model_{iter}.pt mỗi 50 iterations                       │
└─────────────────────────────────────────────────────────────────┘
```

### Toàn bộ Hyperparameters G1 — Tổng hợp

| Parameter | Giá trị | Tại sao |
|-----------|---------|---------|
| **Actor** | MLP [512, 256, 128], ELU | Đủ capacity cho 29-DOF locomotion, giảm dần → compress knowledge |
| **Critic** | MLP [512, 256, 128], ELU | Cùng kiến trúc, nhận thêm privileged info (111 dims vs 99) |
| `obs_normalization` | True | Input range khác nhau → phải chuẩn hóa |
| `learning_rate` | 1e-3 (adaptive) | Bắt đầu nhanh, tự giảm theo KL |
| `desired_kl` | 0.01 | Policy không đổi quá 1% mỗi update |
| `clip_param` | 0.2 | ratio ∈ [0.8, 1.2] — industry standard |
| `gamma` | 0.99 | ~100 steps horizon ≈ 2s tương lai (ở 50Hz) |
| `lam` | 0.95 | GAE: ~20 steps effective — bias-variance balance |
| `num_steps_per_env` | 24 | ~0.48s kinh nghiệm — đủ thấy hệ quả ngã |
| `num_learning_epochs` | 5 | Reuse data 5 lần — on-policy không nên quá nhiều |
| `num_mini_batches` | 4 | Batch 3,072 samples — vừa phải cho GPU 4GB |
| `entropy_coef` | 0.01 | Khuyến khích exploration nhẹ nhàng |
| `value_loss_coef` | 1.0 | Critic quan trọng ngang policy |
| `max_grad_norm` | 1.0 | Chống gradient explosion |
| `max_iterations` | 30,000 | ~7h20m trên RTX 3050 Ti, 512 envs |

---

## Phần IX: Kết nối → Doc tiếp theo

PPO tối ưu hóa **tổng reward**. Nhưng reward thiết kế thế nào ảnh hưởng **trực tiếp** đến hành vi robot:
- Reward sai → robot "cheat" (tìm reward cao mà không đi)  
- Reward thiếu → robot có hành vi không mong muốn (giật, trượt)
- Reward đúng → robot đi tự nhiên, ổn định

→ **Doc tiếp theo** ([reward_engineering.md](reward_engineering.md)) phân tích chi tiết 14 reward terms và tại sao chúng thiết kế như vậy.

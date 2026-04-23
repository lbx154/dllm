# BT-GRPO — 面向 Masked Diffusion LLM 的 Branching-Trajectory GRPO

> 状态：v0 设计文档 + 已落地实现。当前 fork 跑到 run11（见
> [`RUN_HISTORY.md`](RUN_HISTORY.md)），代码见 `dllm/pipelines/rl/btgrpo/`。

## 1. 一句话概括

现有的 masked diffusion LLM 强化学习方法（d1 / diffu-GRPO、GDPO、SPG、
ESPO、Co-GRPO、TIC-GRPO、DiPOD、AWM 等）都把 dLLM 当成 AR 的黑盒替代：
每条 prompt 独立做 **G 次 rollout**，每次给一个 **序列级 scalar
reward**，再用一条 log-prob surrogate 做策略更新。

**这些方法扔掉了什么？** 一条 denoising 轨迹 `τ = (x_T → x_{T-1} → … → x_0)`
会经过一串**部分被 mask 的中间态**，而每一个中间态都是 **合法的、可继续
采样的完整状态**（AR 不具备这个性质，AR 的中间态是前缀而不是状态）。
这意味着我们**可以让多条 rollout 共享轨迹的前半段**，再在后半段分岔。
这是 AR 做不到的，也是 dLLM 在 RL 上最有价值的结构性特征。

**BT-GRPO** 把这个观察落地为训练算法：

1. 对每条 prompt，先把 `x_T → x_{k*}` 这段 denoising **只跑一次**（共享前缀）。
2. 从 `x_{k*}` 开始，**分叉 G 条独立后续**，每条跑到 `x_0^i`。
3. 按常规给每条 `x_0^i` 打分，在 **fork 组内** 做 group-relative advantage。
4. PPO/GRPO 的更新 **只作用在 post-fork（divergent）位置** 上：组内所有
   G 条都一致的位置天然权重为 0。

这样就得到了在 AR 里需要学 critic 或 MCTS 才能拿到的
**per-fork-step credit assignment**；但在 dLLM 里，因为 group baseline 和
共享前缀，**前缀方差被精确抵消**，而且完全 critic-free。

## 2. 理论定位

### 2.1 d1 / diffu-GRPO 的基线做法

给定 prompt q 和 G 条采样 `{o_i}`，d1 算：

```
Â_i = r_i − mean_j r_j
log π_θ(o_i) ≈ 在随机 mask 下的 single-step CE
```

然后套 PPO clip。advantage 里混入的方差有两种来源：

- **(a) prompt 共享方差**："这题难" — 被 `mean_j r_j` 抵消 ✔
- **(b) rollout 轨迹方差**：同一 prompt 下的 G 条 denoising 轨迹不同
  导致 `o_i` 不同 — **不被抵消**，污染 advantage ✗

### 2.2 BT-GRPO 做了什么

让 G 条 fork-兄弟共享前缀 `x_T → x_{k*}`，那 x_{k*} 以前的轨迹方差也会
在 `r_i − r̄` 里被减掉。剩下的只有 `x_{k*} → x_0` 这段的方差，而这段恰好
就是更新想要归因的对象。

严格地说，如果 reward 可拆成 `r_i = f(τ_i_pre) + g(τ_i_post) + ε_i`，
在 BT-GRPO 的共享前缀构造下 `τ_i_pre = τ_pre` 对所有 i 都是确定的，所以：

```
Â_i = g(τ_i_post) − mean_j g(τ_j_post) + ε_i − mean_j ε_j
```

`f` 被完全剥掉。这是 **shared-baseline control variate** 在 diffusion
上的对应物，而且还是"赚到的"：Phase 1 每条 prompt 只跑一次（不是 G 次），
反而省了前向算力，不是多花算力。

### 2.3 和相关工作的比较

| 方法 | off-policy mask | 逐步 credit | 前缀方差抵消 | 用到 dLLM 特有结构 |
|---|---|---|---|---|
| diffu-GRPO (d1) | ❌ | ❌ | ❌ | ❌ |
| TIC-GRPO | ❌ | ❌ | ❌ | ❌ |
| GDPO / SPG / ESPO | ❌ | ❌ | ❌ | ❌ |
| Co-GRPO | ⚠️ (学 scheduler) | ❌ | ❌ | ⚠️ |
| **BT-GRPO（我们）** | ✅ | ✅ | ✅ | ✅ |

最右边那一栏是关键：denoising 中间态 x_k 是 **非因果的、完整的**，从它
分叉出去的 G 条 rollout 共享双向上下文，这点 AR 做不到。

## 3. 算法

```
输入：prompt batch {q_1, …, q_B}，fork_frac f ∈ (0,1)，分支数 G，总步数 T

# Phase 1 —— 共享轨迹（每条 prompt 跑一次）
for each q_b:
    x_b ← FullyMask(q_b, max_new_tokens)
    MDLM denoise 跑 ⌊f·T⌋ 步 → x_b^{(fork)}

# Phase 2 —— 分叉
for each q_b:
    for g = 1…G:
        x_{b,g} ← copy(x_b^{(fork)})
        MDLM denoise 再跑 T − ⌊f·T⌋ 步（独立 RNG）→ o_{b,g}

# Credit 分配
for each q_b:
    r_{b,1}, …, r_{b,G} ← reward_fn(q_b, o_{b,g})
    Â_{b,g} ← r_{b,g} − mean_g r_{b,g}

# 策略更新：梯度只落在 divergent 位置
for each (b, g):
    divergent_mask_{b,g,t} = 1 当且仅当 o_{b,1,t}, …, o_{b,G,t} 不全相同
    loss 贡献 ∝ Â_{b,g} · ρ_{b,g,t} · divergent_mask_{b,g,t}
```

### 为什么 `divergent_mask` 是对的

如果某个位置 t 上 G 条分支产出的 token 完全一致，那 PPO ratio 的分子
分母在组内也一样，它对任何 group-relative advantage 的贡献精确为 0。
显式 mask 掉只是把这个恒等写到明面上，顺便少算一点噪声。

> **实际落地的小坑**：v0 版本默认 `apply_divergent_mask=True`，但在
> LLaDA-8B + GSM8K 上 ~78% 的 completion token 会被它清零，梯度信号
> 骤降。run5 之后我们把它关掉了。见 `RUN_HISTORY.md` run4 的诊断段。

## 4. v0 实现范围

- **单 fork 点**：`k* = ⌊f·T⌋`，f 可配（默认 0.5）。
- `BranchingMDLMSampler` 复用 MDLM step-loop 逻辑，拆成两段、中间传
  state。
- `BTGRPOTrainer` 继承 `DiffuGRPOTrainer`：换 sampler、把
  `completion_mask` 乘上一个 `divergent_mask`。其余（PPO clip、β-KL、
  ref-sync）都从父类继承。
- **不在 v0 里**：多级（递归）分叉、自适应 fork 深度、per-step TR-GRPO
  精确 log-prob。都可以单独做。

## 5. v1：可学习的 `fork_frac`（run6 起）

固定 `fork_frac=0.5` 在实验里发现太粗——简单 prompt 可以早分叉，难的
prompt 反而需要共享更多 CoT。所以我们在 v1 加了一个按 prompt 条件的
**fork head**，在每条 prompt 上采一个 `fork_frac ~ π_ψ(·|h_q)`，由
REINFORCE 更新。

这条路径踩了四个坑（rsample 抵消梯度、dense 4096→1 head 天然不稳、
sigmoid 参数化衰减更新、全局 EMA baseline 让 head 退化成难度分类器），
最终落在下面这个架构：

```
fork head (fp32)
  LayerNorm(4096)
  ├─ Linear(4096 → 8)      共享 bottleneck
  │   ├─ Linear(8 → 1)  →  μ（fork_frac 均值，直接线性参数化）
  │   └─ Linear(8 → 1)  →  V(h)（per-prompt value）
  └─ log_sigma: scalar    →  σ（可学习）

采样： a ~ Normal(μ, σ).sample().detach()，clip 到 [f_min+ε, f_max−ε]
更新： L = −log π(a|h)·(r − V(h).detach()) + (V(h) − r)²
```

- **actor-critic** 而不是 EMA baseline：避免 head 学成"难度检测器"
  （难 prompt → r 低 → `r − EMA(r)` 始终为负 → head 偏向某个
  fork_frac 只是因为那个 fork_frac 常被难题采到，而不是因为它对结果好）。
- **跨 rank all-reduce**：head 的梯度和 value loss 要在所有 rank
  间聚合，保证每卡更新一致。
- 完整故事和逐个 bug 修复见 [`FORK_HEAD.md`](FORK_HEAD.md)。

## 6. 这次 launch 的超参（8×B200 单机）

| 超参 | 当前值（run11） | 说明 |
|---|---|---|
| `fork_frac` | 可学，初始 μ=0.5，范围 `[0.2, 0.8]` | run6 起用 fork head 动态决定 |
| `num_generations` G | 4 | 每卡 1 个 fork 组 |
| `block_size` | 32 | 与 d1 一致 |
| `steps`（denoising） | 64 | |
| `max_completion_length` | 266 | run6 起从 200 提高 |
| `beta` | 0 | k3 KL 在 dLLM 上不稳，关掉 |
| `num_iterations` | 1 | 纯 on-policy，PPO clip 作为唯一 trust region |
| `epsilon` | 0.2 | PPO clip |
| `per_device_train_batch_size` | 4 | = G，保证每卡恰好 1 个 fork 组 |
| `apply_divergent_mask` | **False** | True 会清零 ~78% 的 token 梯度 |
| `scale_rewards` | True | 组内 std 归一化 |
| `learning_rate` | 3e-6 | policy（LoRA） |
| `fork_head_lr` | 1e-3 | fork head 专用 optimizer |
| `LoRA r / α` | 64 / 32 | run2 起 |

有效 batch = 8 GPU × 4 = 32 条 completion/opt step = 8 unique prompt ×
G=4 branch。

## 7. 后续消融计划

1. **BT-GRPO vs d1 同算力**：预期 BT-GRPO 在 sample efficiency 上占优，
   来自前缀方差抵消。
2. **`fork_frac` 扫描**：`{0.25, 0.5, 0.75}` 固定值 vs learned；量化
   credit assignment 的敏感区间。
3. **divergent_mask on vs off**：对理论声明做 sanity check（run5 之后
   我们默认 off，on 掉会扼杀学习）。
4. **BT-GRPO + TR-GRPO-K1 IS 叠加**：两项优化方向正交，预期可叠加。
5. **fork head 消融**：learned fork vs 固定 0.5 vs 固定 0.35；搭配
   value head / EMA baseline 的消融。

## 8. 相关文件

```
dllm/core/samplers/mdlm_branching.py     BranchingMDLMSampler
dllm/pipelines/rl/btgrpo/__init__.py
dllm/pipelines/rl/btgrpo/trainer.py      BTGRPOConfig + BTGRPOTrainer + ForkHead 接入
dllm/pipelines/rl/btgrpo/fork_head.py    ForkHead（LN + bottleneck + π + V）
examples/rl/grpo/llada/train_btgrpo.py   入口
scripts/launch_btgrpo_run{5..11}.sh      历次 run 的启动脚本
docs/BT_GRPO.md                          本文件
docs/FORK_HEAD.md                        fork head 设计 + bug 日记
docs/RUN_HISTORY.md                      逐 run 完整记录
```

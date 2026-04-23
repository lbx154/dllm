# BT-GRPO on LLaDA-8B — 实验日志

本仓库是 `dllm` 的 fork，专门用来做 **Branching-Trajectory GRPO
(BT-GRPO)** 在 `LLaDA-8B-Instruct` + LoRA 上的实验，数据集是 GSM8K。
这个 README 就是实验的主账本：每一次 run 改了什么、为什么改、结果如何。
想看细节的话：

- [`docs/BT_GRPO.md`](docs/BT_GRPO.md) — BT-GRPO 算法本身的设计/理论
- [`docs/RUN_HISTORY.md`](docs/RUN_HISTORY.md) — run1 → run11 的逐次详细记录
- [`docs/FORK_HEAD.md`](docs/FORK_HEAD.md) — 可学习 `fork_frac` head 的设计文档与 bug 日记

---

## 工作流约定（run12 起强制生效）

**每次开新 run 之前先 commit 一次代码。** commit 就是这次 run 对应的源码快照；没有 commit 的话，run 和代码版本的对应关系就丢了。格式：

```
runN: <一行变更摘要>

<可选的多行细节>
```

run1–run4 都有自己的 commit。run5–run11 是在同一个会话里连跑的、没按规矩 commit，所以被合成一个追补 commit `bb83e7f` 锁定 run11 的代码快照。

---

## 硬件 / 基线配置

- 单机 **8× B200**
- 模型：`LLaDA-8B-Instruct` + LoRA（从 run2 起固定 `r=64, α=32`）
- 数据：GSM8K + reasoning prompt；`num_generations G=4`，`per_device_train_batch_size=4` → **每卡恰好 1 个 fork 组**
- Reward 权重（run2 起）：`[xmlcount 0.25, soft_format 0.25, strict_format 0.25, int 0.25, correctness 5.0]`
- 采样：64 denoising steps，`max_completion_length=266`（run6 起）

---

## Run 总览

下表中 reward / correctness 都是该 run 结束前 MA20。"Corr" =
`rewards/correctness_reward_func/mean`，取值 `[0, 2]`。"fork μ" 是 head
采样出的 `fork_frac` 均值（固定 fork 的 run 恒为 0.5）。

| Run     | Commit        | Δ（相对上一 run）                                                                                  | Steps | MA20 loss | MA20 reward | MA20 corr | MA20 grad | fork μ | 结果                                            |
| ------- | ------------- | ------------------------------------------------------------------------------------------------- | ----: | --------: | ----------: | --------: | --------: | -----: | ----------------------------------------------- |
| run1    | `85576bc`     | 初版 BT-GRPO：`num_iter=4, β=0.02, lr=3e-6, eps=0.5, fork=0.5, lora_r=128`                          |   n/a |       n/a |         n/a |  **0.47** |       n/a |    0.5 | 起手 corr 还行，但 grad 会炸                    |
| run2    | `14c1b63`     | `num_iter 4→1, β 0.02→0.04, eps 0.5→0.3, lr 3e-6→1.5e-6, fork 0.5→0.35, lora_r 128→64, corr×5`     |   n/a |       n/a |         n/a |       n/a |       n/a |   0.35 | **KL 爆到 ~1e9**                                |
| run3    | `0be8638`     | 代码侧修补：`kl_ratio_clip=5.0`、`1/f_D` 优势校正、`scale_rewards=True`                             |   n/a |       n/a |         n/a |  **0.18** |       n/a |   0.35 | KL 仍 1e8，β·KL 压过 reward 1000:1              |
| run4    | `bae7590`     | `β 0.04→0`, `fork 0.35→0.5`, `eps 0.3→0.2`, `sync_ref_model=False`                                 | 1500+ |        ~0 |      ~1.05  |    ~0.20  |       low |    0.5 | **学不动，指标全平**                            |
| run5.a1 | `bb83e7f`     | `num_iter=2, β=0.02, lr=1e-5, scale_rewards=F`                                                     |     7 |   **7e8** |      0.88   |    0.17   |   **5e9** |    0.5 | **KL 爆 — 当场杀**                              |
| run5.a2 | `bb83e7f`     | `num_iter=1, β=0, lr=3e-6, scale_rewards=T, kl_ratio_clip=2.0`                                     |    10 |   **1e7** |      1.17   |    0.23   | **8e11**  |    0.5 | 继续炸 — 杀                                     |
| run5    | `bb83e7f`     | 同 a2，但 `apply_divergent_mask=False`，修 per-rank `adv_scale`                                     |   159 |   -0.003  |      1.19   |    0.23   |     0.62  |    0.5 | **稳了；但 correctness 仍平**                   |
| run6    | `bb83e7f`     | 加可学习 fork_frac（sigmoid policy, lr=1e-3）、`max_completion_length=266`                          |    34 |   -0.003  |      0.85   |    0.16   |     0.43  | **0.500** | head 不动                                   |
| run7    | `bb83e7f`     | fp32 ForkHead、改成直接线性参数化、lr=1e-2                                                          |    17 |    0.019  |      1.29   |    0.25   |     0.59  | **0.500** | head 还是死在 0.5                           |
| run8    | `bb83e7f`     | REINFORCE bug：`rsample` → `sample().detach()`                                                     |    11 |    0.022  |      1.19   |    0.23   |     0.45  |  0.745 | μ 动了，但**一步就顶到 0.8**                    |
| run9    | `bb83e7f`     | lr 回到 1e-3                                                                                       |    19 |    0.010  |      1.31   |    0.25   |     0.50  |  0.768 | 仍在第 3 步饱和                                 |
| run10   | `bb83e7f`     | 加 LayerNorm + bottleneck 4096→8 的 ForkHead                                                       |   122 |    0.002  |      0.69   |    0.13   |     0.44  |  0.800 | μ 平滑上升最终饱和；**reward 反而下滑**         |
| run11   | `bb83e7f`     | 加 value head `V(h)`，把 EMA baseline 换成 actor-critic                                             |   243 |    0.002  |      0.90   |  **0.22** |     0.45  |**0.800** | μ 瞬间顶 clamp（新 bug，见 §5.5）；base 学得极慢 |
| run12   | `514731a`     | 深度诊断：`max_comp 266→512`、`num_iter 1→4`、`block 32→64`、`G/bs 4→8`、strict_format 权重 0、xmlcount 去负奖励、**fork head 关** |  跑中 |         — |           — |         — |         — |      — | —                                                |

per-run 的完整动机 / 配置 / 诊断见 `docs/RUN_HISTORY.md`，fork head 的
bug 日记见 `docs/FORK_HEAD.md`。

---

## 贯穿全程的经验教训

1. **dLLM 上只有 `num_iterations=1, β=0` 这一档 GRPO 是稳的。**
   k3 KL 估计 `exp(r) − 1 − r` 在 mask 位上会遇到 `|log π_ref − log π_policy|`
   50+ 的情况，直接 overflow 成 ~1e9；而且第一步的 `old_logps` 还没有，
   PPO ratio 恒为 1，多 iter 也没意义。trust region 完全靠 PPO clip。
2. **`apply_divergent_mask=True` 会直接扼杀学习。** run4 里 ~78% 的
   completion-token 梯度被它清零了。留 False。
3. **per-rank `adv_scale` 必须 all-reduce。** 不 reduce 的话每卡看到的
   scale 不一样，在梯度 reduce 那步会静默引入偏差。
4. **REINFORCE 要用 `sample().detach()`，不能 `rsample()`。** `rsample`
   会把 pathwise 梯度穿过均值，和 `log π · A` 项在解析上相消 → head
   永远不动（run6、run7）。
5. **用 `Linear(4096→1)` 做 REINFORCE head 本质上就是不稳的。** 单步
   Adam 和下一个 prompt 的点积是 `O(lr · √H · |h|)` 量级，无论 lr 多小
   都会顶到 clamp 饱和。必须加 LayerNorm + 低秩 bottleneck `4096→8→1`
   （run10 起）。
6. **用全局 EMA 做 baseline → head 变成"难度分类器"。** `A = r − EMA(r)`
   等于让 head 去学"这题容不容易"，而不是"对这条 prompt，fork_frac 调多少最好"。
   必须用 per-prompt value head `V(h)`（actor-critic，run11 起）。
7. **辅助 head 要保持 fp32。** bf16 权重配 Adam moment 会 underflow。
8. **`strict_format_reward_func` 永远是 0。** 现在有 12.5% 的 reward
   权重被它白占，可以考虑丢掉或重写。

---

## 仓库关键路径

```
dllm/pipelines/rl/btgrpo/
  trainer.py         BT-GRPO 训练器；接入 ForkHead、实现 actor-critic 更新
  fork_head.py       可学习的 per-prompt fork_frac head（LN + 4096→8→{π, V}，fp32）
docs/
  BT_GRPO.md         BT-GRPO 算法本身的设计文档（理论/定位/算法）
  FORK_HEAD.md       fork head 的设计文档 + bug 日记
  RUN_HISTORY.md     逐 run 的完整记录
examples/rl/grpo/llada/
  train_btgrpo.py    入口（TrlParser → BTGRPOConfig）
scripts/
  launch_btgrpo_run{5..11}.sh   历次 run 的启动脚本
dashboard.py         训练实时看板（10 个面板）
monitor.py           日志尾随小工具
```

---

## 使用

**监控一条在跑的 run：**

```bash
python dashboard.py --log .logs/btgrpo-run11.log --refresh 30
```

**开一条新 run（必须先 commit）：**

```bash
git add -A
git commit -m "run12: <delta>"
bash scripts/launch_btgrpo_run12.sh
```

启动脚本会把日志写到 `.logs/btgrpo-run<N>.log`，旧日志统一归档到
`.logs/archive/`。

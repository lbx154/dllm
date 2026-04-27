# BT-GRPO 实验日志（base/dllm fork）

本仓库是 [`ZHZisZZ/dllm`](https://github.com/ZHZisZZ/dllm) 的 fork（push 到
[`lbx154/dllm`](https://github.com/lbx154/dllm) 的 `btgrpo-base-port` 分支），
专门做最小侵入版本的 **Branching-Trajectory GRPO (BT-GRPO)** 在
`LLaDA-8B-Instruct` + LoRA + GSM8K 上的实验。

> 上游 `dllm` 的原 README 在 [`README.upstream.md`](README.upstream.md)。
> 这个 README 是实验主账本：每一次 run 改了什么、为什么改、结果如何。

## 工作流约定

每次开新 run **先 commit**，commit message 格式：

```
runN: <一行变更摘要>

<可选细节>
```

run 编号 = 这个表里的下一个 N（不区分 vanilla/BT，连续递增）。  
commit hash 写到表的 "Commit" 列。

## 硬件 / 基线配置

- 单机 **8× H100 / B200**（每卡 ~180 GB）
- 模型：`GSAI-ML/LLaDA-8B-Instruct` + LoRA（`r=128, α=64, dropout=0.05`，4-bit base）
- 数据：`gsm8k`，`max_prompt_length=` 默认，`max_completion_length=256`
- 采样：`steps=128`，`block_size=32` → 8 blocks × 16 steps/block
- 训练：`max_steps=1000`，`lr=3e-6`，`G=6`，`per_device_bs=6`，`grad_accum=2`，
  `num_iterations=12`，`β=0.04`，`ε=0.5`，`p_mask_prompt=0.15`，
  `sync_ref_model=False`
- DeepSpeed ZeRO-2，`WANDB_MODE=disabled`

## Run 总览

| Run     | Commit    | 算法                  | 关键参数                                                        | Steps | 末段 reward | 末段 KL | 备注                                                              |
| ------- | --------- | --------------------- | --------------------------------------------------------------- | ----: | ----------: | ------: | ----------------------------------------------------------------- |
| vanilla | upstream  | DiffuGRPO（无分叉）   | `MDLMSampler`，无 fork                                           |  1000 | ~baseline   | ~0.4    | 基线，`grpo-base-run1.log`                                        |
| BT1     | `afbfb9f` | BT-GRPO，fork-only    | `fork_frac=0.5`，`apply_divergent_mask=False`                    |  ~440 | —           | 稳      | 中途杀，仅作为 fork-only 对照；shared 占一半步                    |
| BT2     | `f383d12` | BT-GRPO，semi-AR fork | `fork_frac=0.06`，`apply_divergent_mask=True`，semi-AR 模式      |  1000 | 比 vanilla 高一点 | 0.4–0.7 | KL 稳；但发现 mid-block schedule slicing bug → 每条补全漏 ~7 个 mask token，soft_format reward 被 leak 的 mask 砸低 |
| BT3     | `b62bf0d` | BT-GRPO，per-block fork | `fork_frac=0.06`，`per_block_fork=True`，`apply_divergent_mask=True` |  跑中 | —           | —       | **本次重点**。每 block 都跑 1/16 步 shared denoising，anchor 散布全序列；规避 mid-block bug；完整 1000 步对比 |

末段 = 最后 ~50 步 MA。

## 代码改动逐次摘要

### vanilla（upstream）
直接用 `dllm.pipelines.rl.grpo.DiffuGRPOTrainer + MDLMSampler`。
log: `.logs/grpo-base-run1.log`。

### BT1 — `afbfb9f` "Add BT-GRPO minimal port"
- 新增 `dllm/core/samplers/mdlm_branching.py`：`BranchingMDLMSampler`，
  semi-AR fork（fork 落在某个 block 内的某一步），单 phase 1 + tile 到 G branch
  做 phase 2。
- 新增 `dllm/pipelines/rl/btgrpo/{__init__.py,trainer.py}`：
  `BTGRPOTrainer/BTGRPOConfig`。最初带 `1/f_D` advantage scaling，
  `apply_divergent_mask` 默认 True 但通过 cross-sibling 一致性算 mask。
- 新增 `examples/rl/grpo/llada/train_bt.py`：训练入口。

### BT2.a — `3d7fc04` "Drop 1/f_D default, add rollout dump"
- TRL 0.19 GRPO loss 是 mean-normalized over `completion_mask`，1/f_D
  不能补偿被 mask 掉的 token，只是放大有效 LR → KL 爆。改 `apply_adv_scale`
  默认 False。
- 加 rank-0 jsonl rollout dump（`rollout_dump_path`，每 N 步一次）。

### BT2.b — `f383d12` "Mask shared-phase tokens (not cross-sibling agreement)"
- `apply_divergent_mask` 改成消费 sampler-side 的 shared-phase mask：
  `BranchingMDLMSampler` 在 Phase 1 前 snapshot `initial_unmasked1`，Phase 1
  结束后算 `(x1 != mask_id) & attn1.bool() & ~initial_unmasked1`，tile 到
  G branches 存到 `_shared_phase_masks_chunks`。trainer 端读出来 zero 掉
  对应的 `completion_mask`。
- 语义更干净：mask 掉**真正在 shared phase 已确定**的 token，而不是事后
  靠 sibling 一致性反推。

### BT2 诊断（未提交修复）
跑出来发现每条 completion 恰好有 `fork_step_idx = 7` 个 `<|mdm_mask|>`
literal token 漏到答案里 → soft_format reward 暴跌。根因：`_run_blocks`
对 fork 落点那个 block 的 Phase 2，`get_num_transfer_tokens` 用 full
`steps_per_block=16` 重新算 schedule，但只跑 `s_hi - s_lo` 步 → schedule
末尾 ~7 步该解的 mask 没解。**这个 bug 只在 fork 落 mid-block 时发生**。

### BT3 — `b62bf0d` "BT-GRPO: per-block fork (Phase-1 warm-up on every block)"
- `BranchingMDLMSamplerConfig.per_block_fork: bool = True`：Phase 1 在
  **每个 block** 都跑前 `fork_frac` 比例的步（不再是 semi-AR 单点 fork）。
  - 8 blocks × `ceil(0.06 × 16) = 1` 步 = 8 步 shared NFE，分散到全序列
  - Phase 2 每 block 跑剩下 15 步，G branch 各自填
  - 总 NFE 不变（128）
- `_run_blocks` 加 `uniform_fracs` 参数 + `math.ceil` 边界（避免
  `fork_frac` 太小时 round 到 0 步）
- 规避 mid-block schedule slicing bug，因为 Phase 1 / Phase 2 的边界整数
  对齐到 step 数
- 同时保住 diffusion-fork 语义（anchor 散布 vs AR-style prefix fork）

启动脚本：`scripts/bt/launch_bt3.sh`。log: `.logs/grpo-bt3-run1.log`。

## 已知遗留 bug（暂未修）

1. **mask_id 没在 logits 上 suppress** —— 模型可以 argmax 到 mask_id
   token，影响 vanilla/BT 两个 sampler。优先级低，因为 mask_id 通常
   logit 很低。
2. （已通过 BT3 per-block fork **规避**，未"真正"修复）`_run_blocks`
   的 schedule slicing bug 在 legacy semi-AR 模式 + mid-block fork 下仍存在。

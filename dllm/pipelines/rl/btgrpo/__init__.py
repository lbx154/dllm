"""dLLM BT-GRPO pipeline (minimal port).

Adds two corrections on top of vanilla DiffuGRPOTrainer when used with
BranchingMDLMSampler:

1. ``divergent_mask`` — zero out completion_mask at positions where every
   sibling in a fork-group has the same token (those positions contribute
   identically to every branch's log-prob and therefore yield 0 advantage).
2. ``1/f_D`` advantage rescaling — since BT-GRPO only updates on the
   divergent fraction f_D of tokens, multiply advantages by 1/f_D so the
   expected policy-gradient magnitude matches vanilla GRPO.
"""

from .trainer import BTGRPOConfig, BTGRPOTrainer  # noqa: F401

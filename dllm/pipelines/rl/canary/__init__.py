"""Canary-RL: early failure detection for long-horizon RL fine-tuning.

See `docs/CANARY_RL.md` and the methodology document at
`session-state/.../files/canary_rl.md` for the full protocol.

Public API:
    from dllm.pipelines.rl.canary import (
        Signatures,           # 6 failure-signature thresholds (data-driven)
        evaluate_signatures,  # dict-of-step -> dict-of-bool tripwires
        preflight,            # Tier 0 static checks (CLI + library)
        Watcher,              # Tier 1+2 log-tailing watcher
        TrendExtrapolator,    # Tier 3 linear extrapolation
    )
"""
from .signatures import Signatures, evaluate_signatures, summarise_window
from .watcher import Watcher
from .trend import TrendExtrapolator
from .oracle import HealthOracle, Prediction, extract_config_features
from . import preflight

__all__ = [
    "Signatures",
    "evaluate_signatures",
    "summarise_window",
    "Watcher",
    "TrendExtrapolator",
    "HealthOracle",
    "Prediction",
    "extract_config_features",
    "preflight",
]

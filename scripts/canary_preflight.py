"""CLI: Tier 0 preflight on a launch script.

Usage:
    python scripts/canary_preflight.py --launch-script scripts/launch_btgrpo_run20.sh

Exits non-zero on blocking failure.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dllm.pipelines.rl.canary.preflight import _main

if __name__ == "__main__":
    _main()

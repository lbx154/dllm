"""CLI: pre-launch failure prediction.

Examples:
  python scripts/canary_oracle.py --launch-script scripts/launch_btgrpo_run20.sh
  python scripts/canary_oracle.py --launch-script scripts/launch_btgrpo_run19.sh --json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dllm.pipelines.rl.canary.oracle import _main
if __name__ == "__main__":
    _main()

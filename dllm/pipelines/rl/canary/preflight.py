"""Tier 0 — Preflight static checks.

Validate a launch script + reward functions BEFORE burning GPU time.

Catches, in order:
  1. Reward weight imbalance (run17 root cause: corr 5.0 vs format 0.25)
  2. Parser silent bugs (run15 root cause: extract_xml_answer returns whole
     completion when <answer> tag missing)
  3. Reward-function monotonicity violations (run19: lenient parser rewards
     format-collapsed completions higher than well-formatted ones)

Designed to run on CPU in seconds. No model loading.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reward upper bounds, by function name. Used by check_reward_weights.
# Keep in sync with the implementations in dllm/pipelines/rl/grpo/rewards/.
REWARD_MAX = {
    "xmlcount_reward_func":      0.5,   # 4 × 0.125
    "soft_format_reward_func":   0.5,
    "strict_format_reward_func": 0.5,
    "int_reward_func":           0.5,
    "correctness_reward_func":   2.0,
}
# Order in which `--reward_weights` is interpreted by train_btgrpo.py.
REWARD_ORDER = [
    "xmlcount_reward_func",
    "soft_format_reward_func",
    "strict_format_reward_func",
    "int_reward_func",
    "correctness_reward_func",
]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    blocking: bool = True       # if False, only a warning


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok or not c.blocking for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(not c.ok and not c.blocking for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "warnings": self.has_warnings,
            "checks": [c.__dict__ for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def parse_launch_script(path: Path) -> dict:
    """Extract `--key value [value ...]` arguments from a bash launch script."""
    text = path.read_text()
    flat = re.sub(r"\\\n", " ", text)
    # Tokenise; walk left-to-right collecting --flag groups.
    out: dict[str, list[str]] = {}
    toks = flat.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--") and len(t) > 2 and not t.startswith("---"):
            key = t[2:]
            j = i + 1
            vals = []
            while j < len(toks) and not toks[j].startswith("--"):
                # Stop at obvious shell metacharacters (`>>`, `2>&1`, etc.)
                if toks[j] in (">>", ">", "<", "|", "&", "&&", "||", "2>&1"):
                    break
                vals.append(toks[j])
                j += 1
            out[key] = vals
            i = j
        else:
            i += 1
    return out


def check_reward_weights(
    weights: list[float],
    max_ratio: float = 5.0,
) -> CheckResult:
    """Reward-weight imbalance check.

    For each pair (i, j), compute (w_i * R_max_i) / (w_j * R_max_j). If the
    biggest contribution dominates the smallest non-zero one by > max_ratio,
    flag it. Mirrors the run17 diagnosis (corr 27:1 over format).
    """
    if len(weights) != len(REWARD_ORDER):
        return CheckResult(
            "reward_weights",
            ok=False,
            detail=f"expected {len(REWARD_ORDER)} weights, got {len(weights)}",
        )
    contrib = [
        (name, w * REWARD_MAX[name])
        for name, w in zip(REWARD_ORDER, weights)
        if w > 0
    ]
    if not contrib:
        return CheckResult("reward_weights", ok=False,
                           detail="all reward weights are 0")
    contrib_sorted = sorted(contrib, key=lambda x: x[1], reverse=True)
    top_name, top_v = contrib_sorted[0]
    bot_name, bot_v = contrib_sorted[-1]
    ratio = top_v / max(bot_v, 1e-12)
    detail = (
        f"max contribution: {top_name}={top_v:.3f}; "
        f"min: {bot_name}={bot_v:.3f}; ratio={ratio:.1f}× (cap={max_ratio:.1f}×)"
    )
    return CheckResult(
        "reward_weights",
        ok=(ratio <= max_ratio),
        detail=detail,
        blocking=False,   # warning, not abort — historically corr 5:1 was fine
    )


# ---- parser fixtures ------------------------------------------------------
DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "parser_fixtures.json"


def _wrap(text: str) -> list[dict]:
    """Adapt to the trainer's `completions` shape: List[List[Dict]]."""
    return [[{"content": text}]]


def check_parsers(fixtures_path: Path = DEFAULT_FIXTURES) -> list[CheckResult]:
    """Run reward functions on a battery of human-curated fixtures and
    verify each one gets the expected reward bracket."""
    try:
        from dllm.pipelines.rl.grpo.rewards.format import (
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
        )
        from dllm.pipelines.rl.grpo.rewards.math import (
            correctness_reward_func,
            int_reward_func,
        )
    except ImportError as e:
        return [CheckResult("import_rewards", ok=False, detail=str(e))]

    fixtures = json.loads(fixtures_path.read_text())
    results: list[CheckResult] = []
    for fx in fixtures:
        text = fx["completion"]
        gt   = fx["answer"]
        comps = _wrap(text)
        prompts = [[{"role": "user", "content": "dummy"}]]
        # Compute all 5 rewards
        try:
            r_xml    = xmlcount_reward_func(comps)[0]
            r_soft   = soft_format_reward_func(comps)[0]
            r_strict = strict_format_reward_func(comps)[0]
            r_int    = int_reward_func(comps)[0]
            r_corr   = correctness_reward_func(prompts, comps, [gt])[0]
        except Exception as e:
            results.append(CheckResult(f"parser[{fx['id']}]", ok=False, detail=f"{type(e).__name__}: {e}"))
            continue

        actual = {
            "xmlcount": r_xml,
            "soft":     r_soft,
            "strict":   r_strict,
            "int":      r_int,
            "corr":     r_corr,
        }
        # Each fixture specifies expected sign per reward (>0, ==0, "any")
        expects = fx["expects"]
        bad = []
        for k, want in expects.items():
            got = actual[k]
            if want == "pos" and not (got > 0):
                bad.append(f"{k}=expected>0 got {got}")
            elif want == "zero" and not (got == 0):
                bad.append(f"{k}=expected 0 got {got}")
            elif want == "max" and not (abs(got - REWARD_MAX_BY_SHORT[k]) < 1e-6):
                bad.append(f"{k}=expected {REWARD_MAX_BY_SHORT[k]} got {got}")
        results.append(CheckResult(
            f"parser[{fx['id']}: {fx['desc']}]",
            ok=(not bad),
            detail=("; ".join(bad) if bad else
                    f"corr={r_corr} xml={r_xml} soft={r_soft} strict={r_strict} int={r_int}"),
        ))
    return results

REWARD_MAX_BY_SHORT = {"xmlcount": 0.5, "soft": 0.5, "strict": 0.5, "int": 0.5, "corr": 2.0}


def check_monotonicity(fixtures_path: Path = DEFAULT_FIXTURES) -> CheckResult:
    """Verify that 'better' fixtures get higher TOTAL weighted reward than
    'worse' ones, for the default reward weights of run19+. This catches the
    run19 lenient-parser bug at preflight time."""
    fixtures = json.loads(fixtures_path.read_text())
    pairs = [
        # (better_id, worse_id) — each pair specifies a strict ordering we
        # require. 'better' must score >= 'worse'.
        ("good_xml_correct",         "no_tags_lucky_digit"),
        ("good_xml_correct",         "good_xml_wrong_answer"),
        ("good_xml_correct",         "format_collapsed"),
    ]
    fx_by_id = {f["id"]: f for f in fixtures}
    try:
        from dllm.pipelines.rl.grpo.rewards.format import (
            xmlcount_reward_func, soft_format_reward_func,
            strict_format_reward_func,
        )
        from dllm.pipelines.rl.grpo.rewards.math import (
            correctness_reward_func, int_reward_func,
        )
    except ImportError as e:
        return CheckResult("monotonicity", ok=False, detail=str(e))

    # Default run19 weights: xml 1, soft 1, strict 0, int 1, corr 2
    W = [1.0, 1.0, 0.0, 1.0, 2.0]

    def total(text, gt):
        comps = _wrap(text)
        prompts = [[{"role": "user", "content": "x"}]]
        rs = [
            xmlcount_reward_func(comps)[0],
            soft_format_reward_func(comps)[0],
            strict_format_reward_func(comps)[0],
            int_reward_func(comps)[0],
            correctness_reward_func(prompts, comps, [gt])[0],
        ]
        return sum(w * r for w, r in zip(W, rs))

    bad = []
    for better, worse in pairs:
        if better not in fx_by_id or worse not in fx_by_id:
            continue
        b = total(fx_by_id[better]["completion"], fx_by_id[better]["answer"])
        w = total(fx_by_id[worse]["completion"],  fx_by_id[worse]["answer"])
        if b < w:
            bad.append(f"{better}({b:.3f}) < {worse}({w:.3f})")
    return CheckResult(
        "monotonicity",
        ok=(not bad),
        detail=("; ".join(bad) if bad else "all 'better > worse' pairs hold"),
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run_preflight(launch_script: Path | None = None,
                  fixtures: Path = DEFAULT_FIXTURES) -> PreflightReport:
    rep = PreflightReport()

    # Reward weights (need launch script)
    if launch_script is not None:
        try:
            args = parse_launch_script(launch_script)
            rw = args.get("reward_weights", [])
            if rw:
                weights = [float(x) for x in rw]
                rep.checks.append(check_reward_weights(weights))
            else:
                rep.checks.append(CheckResult(
                    "reward_weights", ok=False, blocking=False,
                    detail=f"no --reward_weights in {launch_script}"))
        except Exception as e:
            rep.checks.append(CheckResult(
                "parse_launch_script", ok=False,
                detail=f"{type(e).__name__}: {e}"))

    # Parsers
    rep.checks.extend(check_parsers(fixtures))

    # Monotonicity
    rep.checks.append(check_monotonicity(fixtures))

    return rep


def _main():
    p = argparse.ArgumentParser(description="Tier 0 preflight for BT-GRPO RL runs")
    p.add_argument("--launch-script", type=Path, default=None,
                   help="Path to a launch_btgrpo_runX.sh script")
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES,
                   help="Path to parser_fixtures.json")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of human-readable output")
    args = p.parse_args()

    rep = run_preflight(args.launch_script, args.fixtures)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
        sys.exit(0 if rep.ok else 1)

    GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; RST = "\033[0m"
    for c in rep.checks:
        if c.ok:
            tag, col = "OK  ", GREEN
        elif c.blocking:
            tag, col = "FAIL", RED
        else:
            tag, col = "WARN", YEL
        print(f"  {col}[{tag}]{RST} {c.name}: {c.detail}")
    print()
    if rep.ok:
        print(f"{GREEN}preflight OK{RST}" + (
            f" (with warnings)" if rep.has_warnings else ""))
        sys.exit(0)
    else:
        print(f"{RED}preflight FAILED — fix issues above before launching the run{RST}")
        sys.exit(1)


if __name__ == "__main__":
    _main()

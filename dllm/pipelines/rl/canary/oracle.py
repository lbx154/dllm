"""Health Oracle — pre-launch failure prediction from historical RL traces.

Plug-and-play API:

    from dllm.pipelines.rl.canary import HealthOracle
    oracle = HealthOracle.load_or_train()           # auto-fit on first call
    pred   = oracle.predict("scripts/launch_btgrpo_run21.sh")
    print(pred.summary())

What it does:
1. Parse the launch script's hyperparameters (learning_rate, num_iterations,
   beta, fork_frac, reward_weights, ...) into a numeric feature vector.
2. For each of the 6 canary failure signatures, output P(signature fires
   within first 50 training steps), trained from the 15-run historical
   corpus via leave-one-out logistic regression.
3. Look up the 3 nearest neighbours in normalised feature space and report
   their actual outcomes — gives you analogs to past runs.
4. Generate human-readable recommendations from the predicted dominant
   failure mode and the diff against the closest *healthier* historical run.

The labels come from running each historical run's first 50 steps through
`evaluate_signatures()` — exactly the same code path the live watcher uses,
so "predicted to fire" and "actually fires" mean the same thing.
"""
from __future__ import annotations
import json
import pickle
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from .preflight import parse_launch_script, REWARD_ORDER, REWARD_MAX
from .signatures import evaluate_signatures, Signatures

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_DIR     = Path(__file__).parent
MODEL_PATH  = PKG_DIR / "models" / "oracle.pkl"
DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
DEFAULT_RUNS_CSV    = Path("/root/.copilot/session-state/e0760e9f-dd31-44a0-9a31-10dead3cff60/files/runs_per_step.csv")

# Signatures we predict, one binary classifier per.
SIGNATURE_NAMES = (
    "grad_blowup",
    "starved_signal",
    "fork_saturated",
    "len_collapsing",
    "corr_dead_early",
    "corr_negative_slope",
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _f(args: dict, key: str, default: float, idx: int = 0) -> float:
    v = args.get(key, [])
    if not v:
        return default
    try:
        return float(v[idx])
    except (ValueError, IndexError):
        return default


def _b(args: dict, key: str, default: bool = False) -> float:
    v = args.get(key, [])
    if not v:
        return float(default)
    s = v[0].strip().lower()
    return float(s in ("true", "1", "yes"))


def extract_config_features(args_or_path) -> dict:
    """Extract numeric config features from either a parsed args dict or a
    launch-script Path. Returns a flat dict suitable for feeding into a model."""
    if isinstance(args_or_path, (str, Path)):
        args = parse_launch_script(Path(args_or_path))
    else:
        args = args_or_path

    feat = {}
    feat["learning_rate"]    = _f(args, "learning_rate", 3e-6)
    feat["num_iterations"]   = _f(args, "num_iterations", 1)
    feat["beta"]             = _f(args, "beta", 0.0)
    feat["epsilon"]          = _f(args, "epsilon", 0.2)
    feat["fork_frac"]        = _f(args, "fork_frac", 0.5)
    feat["learn_fork_frac"]  = _b(args, "learn_fork_frac")
    feat["fork_frac_min"]    = _f(args, "fork_frac_min", 0.0)
    feat["fork_frac_max"]    = _f(args, "fork_frac_max", 1.0)
    feat["fork_head_lr"]     = _f(args, "fork_head_lr", 0.0)
    feat["num_generations"]  = _f(args, "num_generations", 4)
    feat["per_device_bs"]    = _f(args, "per_device_train_batch_size", 4)
    feat["max_comp_length"]  = _f(args, "max_completion_length", 256)
    feat["block_size"]       = _f(args, "block_size", 32)
    feat["lora_r"]           = _f(args, "lora_r", 64)
    feat["scale_rewards"]    = _b(args, "scale_rewards")
    feat["filter_zero_std"]  = _b(args, "filter_zero_std_groups")
    feat["filter_zero_correct"] = _b(args, "filter_zero_correct_groups")

    # reward weights, in canonical order from preflight.REWARD_ORDER
    rw = args.get("reward_weights", [])
    rw = [float(x) for x in rw] + [0.0] * (len(REWARD_ORDER) - len(rw))
    for name, w in zip(REWARD_ORDER, rw):
        feat[f"w_{name.split('_')[0]}"] = w

    # ---- derived features ---------------------------------------------
    contrib = [(n, w * REWARD_MAX[n]) for n, w in zip(REWARD_ORDER, rw) if w > 0]
    if contrib:
        cs = sorted(c[1] for c in contrib)
        feat["reward_ratio_max_over_min"] = cs[-1] / max(cs[0], 1e-12)
    else:
        feat["reward_ratio_max_over_min"] = 1.0

    feat["grads_per_step"] = feat["num_iterations"] * feat["num_generations"]
    feat["fork_range"]     = feat["fork_frac_max"] - feat["fork_frac_min"]
    return feat


# ---------------------------------------------------------------------------
# Build labels from per-step CSV
# ---------------------------------------------------------------------------
def _labels_for_run(rows: list[dict], n: int = 50) -> dict:
    """Run signatures on first n steps; return {signature: 0/1}."""
    ev = evaluate_signatures(rows[:n], Signatures())
    out = {k: int(ev["advisory"].get(k, False)) for k in SIGNATURE_NAMES}
    out["n_fired"] = int(ev["advisory"].get("n_fired", 0))
    out["any_fired"] = int(out["n_fired"] > 0)
    return out


def _load_run_rows(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path, low_memory=False)
    out = {}
    for run, sub in df.groupby("run"):
        sub = sub.sort_values("step")
        out[run] = [
            {k: v for k, v in r.items() if pd.notna(v) and k != "run"}
            for _, r in sub.iterrows()
        ]
    return out


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    run_tag: str
    p_failure_overall: float
    per_signature_p:   dict[str, float]
    neighbors:         list[tuple[str, float, dict]]   # (run, distance, labels)
    recommendations:   list[str]
    config_features:   dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = []
        lines.append(f"=== HealthOracle prediction for {self.run_tag} ===")
        lines.append(f"  P(failure within 50 steps) = {self.p_failure_overall:.2f}")
        lines.append("  Per-signature probabilities:")
        for k, p in sorted(self.per_signature_p.items(), key=lambda x: -x[1]):
            bar = "█" * int(round(p * 20))
            lines.append(f"    {k:<22s} {p:.2f}  {bar}")
        lines.append("  Nearest historical analogs:")
        for run, dist, labs in self.neighbors:
            fired = [k for k in SIGNATURE_NAMES if labs.get(k)]
            lines.append(f"    {run:<8s} dist={dist:.2f}  fired: {fired or 'none'}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The Oracle
# ---------------------------------------------------------------------------
class HealthOracle:
    def __init__(self):
        self.feature_keys: list[str] | None = None
        self.scaler:  StandardScaler | None = None
        self.models:  dict[str, LogisticRegression] = {}
        self.knn:     NearestNeighbors | None = None
        self.train_runs:    list[str] = []
        self.train_labels:  dict[str, dict] = {}     # run -> {signature: 0/1, ...}
        self.train_features: dict[str, dict] = {}    # run -> feature dict
        self._loo_self_predictions: dict[str, dict] = {}  # diagnostic

    # ---------------- training ----------------------------------------
    def fit(self,
            scripts_dir: Path = DEFAULT_SCRIPTS_DIR,
            runs_csv:    Path = DEFAULT_RUNS_CSV) -> "HealthOracle":
        scripts_dir = Path(scripts_dir)
        runs_csv = Path(runs_csv)

        run_rows = _load_run_rows(runs_csv)

        # collect (run, config_features, signature_labels) tuples
        scripts = sorted(scripts_dir.glob("launch_btgrpo_run*.sh"),
                         key=lambda p: int(re.search(r"run(\d+)", p.stem).group(1)))
        train_X: list[dict] = []
        train_runs: list[str] = []
        train_Y: list[dict] = []
        for sp in scripts:
            m = re.search(r"(run\d+)", sp.stem)
            if not m:
                continue
            run = m.group(1)
            if run not in run_rows or len(run_rows[run]) < 5:
                continue        # no log -> can't label
            feat = extract_config_features(sp)
            labels = _labels_for_run(run_rows[run])
            train_X.append(feat)
            train_runs.append(run)
            train_Y.append(labels)

        if len(train_X) < 4:
            raise RuntimeError(f"too few labelled runs ({len(train_X)}) — need at least 4")

        keys = sorted(train_X[0].keys())
        X = np.array([[fx[k] for k in keys] for fx in train_X], dtype=float)

        self.feature_keys = keys
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        # --- per-signature logistic with LOO self-prediction -------
        for sig in SIGNATURE_NAMES:
            y = np.array([t[sig] for t in train_Y], dtype=int)
            if y.sum() == 0:
                self.models[sig] = None       # never observed → predict 0
                continue
            if y.sum() == len(y):
                self.models[sig] = None       # always observed — degenerate
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     class_weight="balanced").fit(Xs, y)
            self.models[sig] = clf

        # --- LOO predictions for diagnostics -----------------------
        loo = LeaveOneOut()
        for sig in SIGNATURE_NAMES:
            y = np.array([t[sig] for t in train_Y], dtype=int)
            if self.models[sig] is None:
                continue
            preds = np.zeros(len(y))
            for tr, te in loo.split(Xs):
                if y[tr].sum() in (0, len(tr)):
                    preds[te[0]] = float(y[tr].mean())
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0,
                                         class_weight="balanced").fit(Xs[tr], y[tr])
                preds[te[0]] = clf.predict_proba(Xs[te])[0, 1]
            for run, p in zip(train_runs, preds):
                self._loo_self_predictions.setdefault(run, {})[sig] = float(p)

        # --- kNN over normalised features --------------------------
        self.knn = NearestNeighbors(n_neighbors=min(3, len(Xs))).fit(Xs)

        self.train_runs = train_runs
        self.train_features = {r: f for r, f in zip(train_runs, train_X)}
        self.train_labels   = {r: l for r, l in zip(train_runs, train_Y)}
        return self

    # ---------------- predict ------------------------------------------
    def predict(self, script_path: str | Path,
                run_tag: str | None = None) -> Prediction:
        script_path = Path(script_path)
        feat = extract_config_features(script_path)
        run_tag = run_tag or script_path.stem.replace("launch_btgrpo_", "")

        x = np.array([[feat[k] for k in self.feature_keys]], dtype=float)
        xs = self.scaler.transform(x)

        per_sig = {}
        for sig in SIGNATURE_NAMES:
            clf = self.models.get(sig)
            if clf is None:
                base = float(np.mean([self.train_labels[r][sig]
                                      for r in self.train_runs]))
                per_sig[sig] = base
            else:
                per_sig[sig] = float(clf.predict_proba(xs)[0, 1])

        # overall = 1 - product of (1 - p_i) under independence proxy
        p_fail = 1.0 - float(np.prod([1 - p for p in per_sig.values()]))

        # nearest neighbours
        dist, idx = self.knn.kneighbors(xs, return_distance=True)
        neighbors = []
        for d, i in zip(dist[0], idx[0]):
            r = self.train_runs[i]
            neighbors.append((r, float(d), self.train_labels[r]))

        recs = self._recommend(feat, per_sig, neighbors)

        return Prediction(
            run_tag=run_tag,
            p_failure_overall=p_fail,
            per_signature_p=per_sig,
            neighbors=neighbors,
            recommendations=recs,
            config_features=feat,
        )

    # ---------------- recommendations ----------------------------------
    def _recommend(self, feat: dict, per_sig: dict,
                   neighbors: list[tuple[str, float, dict]]) -> list[str]:
        recs = []
        # threshold for "this is likely to fire"
        T = 0.5

        if per_sig.get("grad_blowup", 0) >= T:
            if feat["num_iterations"] > 1:
                recs.append(f"Predicted grad explosion: lower num_iterations "
                            f"{int(feat['num_iterations'])} -> 1 "
                            f"(run12/run13 had num_iter≥2 + ε=0.2 → 80k grad spikes).")
            if feat["learning_rate"] > 5e-6:
                recs.append(f"Predicted grad explosion: lower learning_rate "
                            f"{feat['learning_rate']:.0e} -> 3e-6 or lower.")
            if feat["beta"] == 0:
                recs.append("Predicted grad explosion: consider beta>0 (KL anchor) "
                            "to bound ratio drift.")

        if per_sig.get("starved_signal", 0) >= T:
            if not feat["filter_zero_correct"]:
                recs.append("Predicted signal starvation: enable "
                            "--filter_zero_correct_groups True.")
            if feat["scale_rewards"]:
                recs.append("Predicted signal starvation: disable "
                            "--scale_rewards (run15 root cause: noise amplifier).")
            if feat["num_generations"] < 8:
                recs.append(f"Predicted signal starvation: increase num_generations "
                            f"{int(feat['num_generations'])} -> 8 to reduce zero-std groups.")

        if per_sig.get("fork_saturated", 0) >= T:
            if feat["learn_fork_frac"] and feat["fork_range"] > 0.7:
                recs.append("Predicted fork saturation: tighten fork_frac range "
                            "(e.g. min=0.2, max=0.6).")
            if feat["learn_fork_frac"] and feat["fork_head_lr"] > 5e-4:
                recs.append(f"Predicted fork saturation: lower fork_head_lr "
                            f"{feat['fork_head_lr']:.0e} -> 1e-4.")

        if per_sig.get("len_collapsing", 0) >= T:
            if feat.get("w_correctness_reward_func", feat.get("w_correctness", 0)) > 3.0:
                recs.append("Predicted length collapse: lower correctness weight or "
                            "raise format weights to balance reward hacking pressure.")

        if per_sig.get("corr_dead_early", 0) >= T:
            recs.append("Predicted correctness dead: verify reward parsers on a "
                        "rollout dump (run15 lesson: silent extractor bug).")

        # Reward weight imbalance — always check
        if feat.get("reward_ratio_max_over_min", 1.0) > 5:
            recs.append(f"Reward weight ratio {feat['reward_ratio_max_over_min']:.1f}× "
                        f"(>5×): consider rebalancing — run17 collapsed at 27×, "
                        "current run20 sits at 8×.")

        # Diff against closest healthier neighbour
        if neighbors:
            healthier = [n for n in neighbors
                         if sum(n[2][k] for k in SIGNATURE_NAMES)
                         <  sum(per_sig[k] >= T for k in SIGNATURE_NAMES)]
            if healthier:
                run, _, _ = healthier[0]
                recs.append(f"Closest historical run with FEWER predicted failures: "
                            f"{run} — see scripts/launch_btgrpo_{run}.sh for diff.")

        if not recs:
            recs.append("No high-priority recommendations: all signatures predicted < 0.5.")
        return recs

    # ---------------- persistence --------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "HealthOracle":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_or_train(cls,
                      path: Path = MODEL_PATH,
                      scripts_dir: Path = DEFAULT_SCRIPTS_DIR,
                      runs_csv:    Path = DEFAULT_RUNS_CSV,
                      force: bool = False) -> "HealthOracle":
        if path.exists() and not force:
            try:
                return cls.load(path)
            except Exception:
                pass
        oracle = cls().fit(scripts_dir, runs_csv)
        oracle.save(path)
        return oracle


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Pre-launch RL health oracle")
    ap.add_argument("--launch-script", required=True, type=Path)
    ap.add_argument("--retrain", action="store_true",
                    help="Force re-training from history before predicting")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS_DIR)
    ap.add_argument("--runs-csv",    type=Path, default=DEFAULT_RUNS_CSV)
    args = ap.parse_args()

    oracle = HealthOracle.load_or_train(scripts_dir=args.scripts_dir,
                                        runs_csv=args.runs_csv,
                                        force=args.retrain)
    pred = oracle.predict(args.launch_script)
    if args.json:
        # neighbors contain dicts; make them JSON-friendly
        d = pred.to_dict()
        d["neighbors"] = [list(n) for n in d["neighbors"]]
        print(json.dumps(d, indent=2, default=float))
    else:
        print(pred.summary())


if __name__ == "__main__":
    _main()

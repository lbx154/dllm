"""Tier 1 + Tier 2 log-tailing watcher.

Reads a `.logs/btgrpo-runX.log` file (or any stream of stringified Python
dicts), parses each step into a metric dict, and continuously evaluates the
failure signatures.

Two operating modes:

  - one-shot replay: `Watcher(path).replay()` returns the final state. Used
    in tests and for retrospective analysis of historical runs.

  - live tail: `Watcher(path).follow()` blocks, re-reading the file as it
    grows, yielding a state dict after each new step. Sends SIGINT to a
    target PID when an abort condition fires.
"""
from __future__ import annotations
import ast
import json
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from .signatures import Signatures, evaluate_signatures

DICT_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


# ---------------------------------------------------------------------------
@dataclass
class WatcherState:
    n_steps: int = 0
    rows: list[dict] = field(default_factory=list)
    last_eval: dict | None = None
    aborted: bool = False
    abort_reason: str | None = None


# ---------------------------------------------------------------------------
class Watcher:
    """Stateful log watcher.

    Args
    ----
    path:        log file to read.
    sigs:        threshold container (default Signatures()).
    target_pid:  PID to send SIGINT to on abort. None = print only.
    on_abort:    optional callback(reason: str) invoked instead of SIGINT.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        sigs: Signatures | None = None,
        target_pid: Optional[int] = None,
        on_abort=None,
    ):
        self.path = Path(path)
        self.sigs = sigs or Signatures()
        self.target_pid = target_pid
        self.on_abort = on_abort
        self.state = WatcherState()
        # rolling window for Tier 2 signatures
        self._window = deque(maxlen=self.sigs.window_size)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_dict_lines(text: str) -> list[dict]:
        """Extract every `{...'loss'...}` dict from a chunk of log text."""
        out = []
        for m in DICT_RE.finditer(text):
            try:
                d = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
            if isinstance(d, dict) and "loss" in d:
                out.append(d)
        return out

    # ------------------------------------------------------------------
    def _push(self, row: dict) -> dict:
        self.state.n_steps += 1
        self.state.rows.append(row)
        self._window.append(row)
        ev = evaluate_signatures(list(self._window), self.sigs)
        ev["step"] = self.state.n_steps
        self.state.last_eval = ev

        # Tier 1: hard abort
        for k, v in ev["abort"].items():
            if v and not self.state.aborted:
                self.state.aborted = True
                self.state.abort_reason = k
                self._fire_abort(k, ev)
                break
        return ev

    def _fire_abort(self, reason: str, ev: dict) -> None:
        msg = f"[canary] ABORT: {reason} at step {self.state.n_steps}"
        sys.stderr.write(msg + "\n")
        if self.on_abort:
            self.on_abort(reason)
        elif self.target_pid:
            try:
                os.kill(self.target_pid, signal.SIGINT)
            except ProcessLookupError:
                sys.stderr.write(f"[canary] target pid {self.target_pid} gone\n")

    # ------------------------------------------------------------------
    def replay(self) -> WatcherState:
        """One-shot: parse the entire file, evaluate signatures step by step,
        then return final state."""
        text = self.path.read_text(errors="ignore")
        for row in self.parse_dict_lines(text):
            self._push(row)
            if self.state.aborted:
                break
        return self.state

    def replay_rows(self, rows: Sequence[dict]) -> WatcherState:
        """Same as replay() but on a pre-parsed list of step dicts. Used by
        tests that read from runs_per_step.csv."""
        for row in rows:
            self._push(row)
            if self.state.aborted:
                break
        return self.state

    # ------------------------------------------------------------------
    def follow(self, poll: float = 2.0) -> Iterator[dict]:
        """tail -F equivalent. Yields an evaluation dict after each new step.
        Stops yielding once aborted (caller should break)."""
        if not self.path.exists():
            self.path.touch()
        with self.path.open(errors="ignore") as f:
            buf = ""
            while True:
                chunk = f.read()
                if chunk:
                    buf += chunk
                    # Pull out as many complete dicts as possible; keep tail.
                    last_end = 0
                    for m in DICT_RE.finditer(buf):
                        try:
                            d = ast.literal_eval(m.group(0))
                        except (ValueError, SyntaxError):
                            continue
                        if isinstance(d, dict) and "loss" in d:
                            ev = self._push(d)
                            yield ev
                            last_end = m.end()
                    buf = buf[last_end:]
                    if self.state.aborted:
                        return
                else:
                    time.sleep(poll)

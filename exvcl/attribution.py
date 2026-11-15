"""exvcl.attribution -- the attribution protocol in code: evidence controls and the quantities compared.

Run the SAME gate, stream, budget and seeds with real evidence and with the controls, then compare admitted-poison share, clean-data
retention and downstream harm. Protection that survives uninformative evidence does not come from the evidence content.

    real           the observation's own evidence
    shuffled       evidence permuted among the observations of the same verification batch (keeps routing and batch base rate)
    global_random  evidence drawn at random from a pool collected during calibration (independent of the observation and its class)
    mix            each observation keeps its real evidence with probability q, otherwise global-random (q=1 real, q=0 null)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from evcl.evaluation import holm, paired_stats

CONTROLS = ("real", "shuffled", "global_random", "mix")


class ChunkVerifier:
    """Callable `verify(idx) -> (F, ok)` over one chunk of per-observation evidence F (n, 7), with an evidence control."""

    def __init__(self, F, ok=None, control: str = "real", pool=None, q: float = 1.0, rng: Optional[np.random.Generator] = None):
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")
        self.F = np.asarray(F, float)
        self.ok = np.ones(len(self.F), bool) if ok is None else np.asarray(ok, bool)
        self.control, self.q = control, float(q)
        self.pool = None if pool is None else np.asarray(pool, float)
        if control in ("global_random", "mix") and (self.pool is None or len(self.pool) == 0):
            raise ValueError("global_random and mix controls need `pool`: evidence collected during calibration")
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def __call__(self, idx):
        idx = np.asarray(idx, int)
        F, ok = self.F[idx].copy(), self.ok[idx].copy()
        if self.control == "shuffled":
            p = self.rng.permutation(len(idx))
            F, ok = F[p], ok[p]
        elif self.control == "global_random":
            F = self.pool[self.rng.integers(0, len(self.pool), len(idx))]
        elif self.control == "mix":
            swap = self.rng.random(len(idx)) >= self.q
            if swap.any():
                F[swap] = self.pool[self.rng.integers(0, len(self.pool), int(swap.sum()))]
        return F, ok


def admitted_poison_share(admit, poisoned) -> float:
    """Share of poisoned observations that the gate admitted (lower is better). NaN if there is no poison."""
    admit, poisoned = np.asarray(admit, bool), np.asarray(poisoned, bool)
    return float(admit[poisoned].mean()) if poisoned.any() else float("nan")


def clean_retention(admit, poisoned) -> float:
    """Share of clean observations that the gate admitted (higher is better)."""
    admit, poisoned = np.asarray(admit, bool), np.asarray(poisoned, bool)
    return float(admit[~poisoned].mean()) if (~poisoned).any() else float("nan")


def paired_report(real, control, label: str = "control") -> dict:
    """Paired comparison of a per-seed metric under real evidence vs a control (Wilcoxon p, bootstrap CI of the mean difference).
    Use one value per independent seed. Apply holm() over all comparisons you report."""
    s = paired_stats(np.asarray(control, float), np.asarray(real, float))
    s["comparison"] = f"{label} minus real"
    return s


__all__ = ["CONTROLS", "ChunkVerifier", "admitted_poison_share", "clean_retention", "paired_report", "holm"]

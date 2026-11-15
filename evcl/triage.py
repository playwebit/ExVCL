"""evcl.triage -- userspace model-based triage (NOT a kernel data plane; see sandbox.XDPTriageAdapter)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriageResult:
    score: np.ndarray           # P(attack) = 1 - P(benign) from the current NIDS model
    verify_needed: np.ndarray   # bool: must be verified before it can be admitted on the evidence path


class ModelTriage:
    """Fast path only for confident-normal traffic whose observed label is also benign.
    Anything the model finds suspicious, OR where model and label disagree, needs verification."""

    def __init__(self, low_threshold: float = 0.2, mode: str = "model"):
        if mode not in ("model", "all"):
            raise ValueError(mode)
        self.low, self.mode = low_threshold, mode

    def route(self, X, y_obs, learner) -> TriageResult:
        score = learner.p_attack(X)
        if self.mode == "all":
            need = np.ones(len(score), bool)
        else:
            need = (score >= self.low) | (y_obs != 0)
        return TriageResult(score, need)
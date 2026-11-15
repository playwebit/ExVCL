"""evcl.evidence -- deterministic evidence fusion: telemetry features -> verdict."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

V_BENIGN, V_MALICIOUS, V_UNCERTAIN, V_UNAVAILABLE = 0, 1, 2, 3
VERDICT_NAMES = {0: "benign", 1: "malicious", 2: "uncertain", 3: "unavailable"}


@dataclass
class EvidenceBatch:
    p_malicious: np.ndarray
    verdict: np.ndarray


class CalibratedFusion:
    """Logistic regression on log1p(raw counts), fit on a TRUSTED labelled calibration corpus."""

    def __init__(self, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)

    def fit(self, F, y):
        self.m.fit(np.log1p(F), y)
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(np.log1p(F))[:, 1]


class RuleBasedFusion:
    """Transparent indicator count. Thresholds are ILLUSTRATIVE hand-set values, not validated."""
    THRESH = dict(process_spawn=4, shell_exec=1, file_write=8, sensitive_file_access=1,
                  net_connect_out=6, priv_change=1)

    def fit(self, F, y):
        return self

    def predict_proba(self, F):
        from .telemetry import EVENT_KINDS
        hit = np.zeros(len(F))
        for k, th in self.THRESH.items():
            hit += (F[:, EVENT_KINDS.index(k)] >= th)
        return np.minimum(1.0, hit / 3.0)


class EvidenceFuser:
    def __init__(self, fusion, t_benign=0.2, t_malicious=0.8):
        self.fusion, self.tb, self.tm = fusion, t_benign, t_malicious

    def fuse(self, F: np.ndarray, ok: np.ndarray) -> EvidenceBatch:
        n = len(ok)
        p = np.full(n, np.nan)
        verdict = np.full(n, V_UNAVAILABLE, np.int8)
        if n and ok.any():
            p[ok] = self.fusion.predict_proba(F[ok])
            v = np.full(ok.sum(), V_UNCERTAIN, np.int8)
            v[p[ok] <= self.tb] = V_BENIGN
            v[p[ok] >= self.tm] = V_MALICIOUS
            verdict[ok] = v
        return EvidenceBatch(p, verdict)


def calibrate_thresholds(p_cal: np.ndarray, y_cal_malicious: np.ndarray, target_fpr: float, target_fnr: float,
                         fallback=(0.2, 0.8)):
    """Evidence thresholds from CALIBRATION-period telemetry only.
      t_malicious : smallest t with calibration-benign fraction (p >= t) <= target_fpr
      t_benign    : largest  t with calibration-malicious fraction (p <= t) <= target_fnr
    If the classes overlap (t_benign < t_malicious) the band between them is the 'uncertain' zone (both error targets
    hold). If they are separable (t_benign >= t_malicious) any boundary inside the gap meets both targets, so a single
    midpoint boundary is used (no uncertain zone). Falls back (flagged) only when calibration data are too scarce."""
    pb, pm = p_cal[~y_cal_malicious], p_cal[y_cal_malicious]
    if len(pb) < 10 or len(pm) < 10:
        return (*fallback, "FALLBACK: too few calibration executions")
    t_mal = float(np.quantile(pb, 1 - target_fpr))
    t_ben = float(np.quantile(pm, target_fnr))
    if t_ben < t_mal:
        return t_ben, t_mal, "calibrated (overlap: uncertain zone kept)"
    mid = (t_ben + t_mal) / 2.0
    return mid, mid, "calibrated (separable: single boundary, no uncertain zone)"
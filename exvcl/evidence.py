"""exvcl.evidence -- calibrated evidence model: seven telemetry counts -> verdict (benign / malicious / uncertain / unavailable).

Fitted ONLY on a trusted, labelled calibration set of executions (never on the stream the gate protects). Optional support check
(modification M1 of the paper): evidence that resembles neither the known-benign nor the known-malicious calibration executions is
reported as UNAVAILABLE, so the gate defers it instead of reading a quiet execution as benign.
"""
from __future__ import annotations

import json
from enum import IntEnum
from typing import Optional

import numpy as np

from evcl.evidence import (CalibratedFusion, EvidenceBatch, EvidenceFuser, V_BENIGN, V_MALICIOUS, V_UNAVAILABLE, V_UNCERTAIN,
                           calibrate_thresholds)
from evcl.phase2 import SupportModel
from evcl.telemetry import EVENT_KINDS


class Verdict(IntEnum):
    BENIGN = V_BENIGN
    MALICIOUS = V_MALICIOUS
    UNCERTAIN = V_UNCERTAIN
    UNAVAILABLE = V_UNAVAILABLE


class CalibrationError(ValueError):
    """The calibration corpus is too small or one-sided to fit the evidence model (fail-safe: no evidence, no admission)."""


class EvidenceModel:
    """Logistic fusion on log1p(counts) + thresholds from the calibration period + optional support check (M1)."""

    def __init__(self, target_fpr: float = 0.02, target_fnr: float = 0.02, support_check: bool = False,
                 support_k: int = 3, support_q: float = 0.99, seed: int = 0, min_per_class: int = 10, drop=()):
        self.target_fpr, self.target_fnr = float(target_fpr), float(target_fnr)
        self.support_check, self.support_k, self.support_q = bool(support_check), int(support_k), float(support_q)
        self.seed, self.min_per_class = int(seed), int(min_per_class)
        bad = [k for k in drop if k not in EVENT_KINDS]
        if bad:
            raise ValueError(f"drop must name evidence counts from {EVENT_KINDS}; got {bad}")
        self.drop = tuple(drop)                     # nuisance counts (e.g. 'syscall_total') excluded from fusion and the support check
        self.fusion: Optional[CalibratedFusion] = None
        self.support: Optional[SupportModel] = None
        self.t_benign = self.t_malicious = float("nan")
        self.status = "not fitted"

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _check(F, name="F"):
        F = np.asarray(F, float)
        if F.ndim != 2 or F.shape[1] != len(EVENT_KINDS):
            raise ValueError(f"{name} must have shape (n, {len(EVENT_KINDS)}) in the order {EVENT_KINDS}; got {F.shape}")
        if not np.isfinite(F).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return F

    def _prep(self, F):
        F = self._check(F).copy()
        for k in self.drop:
            F[:, EVENT_KINDS.index(k)] = 0.0
        return F

    def fit(self, F, y_malicious) -> "EvidenceModel":
        F = self._prep(F)
        y = np.asarray(y_malicious).astype(int)
        if len(y) != len(F):
            raise ValueError("F and y_malicious must have the same length")
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        if min(n0, n1) < self.min_per_class:
            raise CalibrationError(f"calibration needs at least {self.min_per_class} benign and {self.min_per_class} malicious "
                                   f"executions (got {n0} benign, {n1} malicious)")
        self.fusion = CalibratedFusion(self.seed).fit(F, y)
        tb, tm, st = calibrate_thresholds(self.fusion.predict_proba(F), y == 1, self.target_fpr, self.target_fnr)
        if st.startswith("FALLBACK"):
            raise CalibrationError(st)
        self.t_benign, self.t_malicious, self.status = float(tb), float(tm), st
        self.support = SupportModel(self.support_k, self.support_q).fit(F, y) if self.support_check else None
        return self

    # ------------------------------------------------------------------ inference
    @property
    def fitted(self) -> bool:
        return self.fusion is not None

    def p_malicious(self, F) -> np.ndarray:
        self._need_fit()
        return self.fusion.predict_proba(self._prep(F))

    def verdicts(self, F, ok=None) -> EvidenceBatch:
        """Evidence rows -> EvidenceBatch(p_malicious, verdict). `ok` marks executions that produced telemetry (False -> UNAVAILABLE).
        With the support check on, unfamiliar evidence is also UNAVAILABLE."""
        self._need_fit()
        F = self._prep(F)
        ok = np.ones(len(F), bool) if ok is None else np.asarray(ok, bool).copy()
        if len(ok) != len(F):
            raise ValueError("ok must have one entry per evidence row")
        if self.support is not None and len(F):
            ok &= ~self.support.unfamiliar(F)
        return EvidenceFuser(self.fusion, self.t_benign, self.t_malicious).fuse(F, ok)

    def _need_fit(self):
        if not self.fitted:
            raise CalibrationError("EvidenceModel is not fitted: call fit(F_cal, y_cal_malicious) first")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        self._need_fit()
        m = self.fusion.m
        d = dict(format=1, event_kinds=list(EVENT_KINDS), target_fpr=self.target_fpr, target_fnr=self.target_fnr,
                 support_check=self.support_check, support_k=self.support_k, support_q=self.support_q, seed=self.seed,
                 min_per_class=self.min_per_class, drop=list(self.drop), t_benign=self.t_benign, t_malicious=self.t_malicious, status=self.status,
                 coef=m.coef_.tolist(), intercept=m.intercept_.tolist(), classes=m.classes_.tolist(), support=None)
        if self.support is not None:
            s = self.support
            d["support"] = dict(k=s.k, q=s.q, thr={str(c): float(s.thr[c]) for c in (0, 1)}, mu=s.mu.tolist(), sd=s.sd.tolist(),
                                ref={str(c): (None if s.ref[c] is None else s.ref[c].tolist()) for c in (0, 1)})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceModel":
        if d.get("format") != 1 or list(d.get("event_kinds", [])) != list(EVENT_KINDS):
            raise ValueError("unsupported or incompatible EvidenceModel file")
        from sklearn.linear_model import LogisticRegression
        e = cls(d["target_fpr"], d["target_fnr"], d["support_check"], d["support_k"], d["support_q"], d["seed"], d["min_per_class"], d.get("drop", ()))
        e.fusion = CalibratedFusion(e.seed)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=e.seed)
        lr.coef_, lr.intercept_ = np.array(d["coef"], float), np.array(d["intercept"], float)
        lr.classes_, lr.n_iter_ = np.array(d["classes"]), np.array([0])
        e.fusion.m = lr
        e.t_benign, e.t_malicious, e.status = d["t_benign"], d["t_malicious"], d["status"]
        if d["support"] is not None:
            s, sd = SupportModel(d["support"]["k"], d["support"]["q"]), d["support"]
            s.thr = {int(c): float(v) for c, v in sd["thr"].items()}
            s.ref = {int(c): (None if v is None else np.array(v, float)) for c, v in sd["ref"].items()}
            s.mu, s.sd = np.array(sd["mu"], float), np.array(sd["sd"], float)
            e.support = s
        return e

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "EvidenceModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def describe(self) -> dict:
        return dict(fitted=self.fitted, t_benign=self.t_benign, t_malicious=self.t_malicious, threshold_status=self.status,
                    support_check=self.support_check)

"""exvcl.gate -- the learning admission gate (modes none, selective, strict).

The gate is independent of the learner: it needs the observed labels, an evidence supplier (`verify`) and, in selective mode, the
current model's attack scores for triage. It returns an admit mask and the label to learn from. It never sees ground truth.

    none       admit everything (accept-all)
    selective  verify only ambiguous observations (not confident-normal, or model and label disagree); the fast path admits the rest,
               with audit slots reserved out of the verification budget
    strict     verify everything; no fast path; observations beyond the budget are deferred and not learned
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from evcl.admission import (AdmitAllPolicy, EvidenceConsistencyPolicy, REASON_NAMES, ROUTE_DEFERRED, ROUTE_FAST, ROUTE_VERIFIED,
                            R_ADMIT_ALL, check_gate_invariant)
from evcl.evidence import V_UNAVAILABLE
from evcl.telemetry import EVENT_KINDS

from .evidence import EvidenceModel

MODES = ("none", "selective", "strict")
VerifyFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class GateResult:
    admit: np.ndarray        # bool, learn from this observation
    label: np.ndarray        # label to learn from (== observed label unless relabelling is on)
    reason: np.ndarray       # int code, see REASON_NAMES
    route: np.ndarray        # 0 fast path, 1 verified, 2 deferred (budget exhausted)
    verdict: np.ndarray      # evidence verdict per observation (3 = unavailable / not verified)
    n_verified: int = 0
    n_audit: int = 0
    n_deferred: int = 0
    n_failed: int = 0

    def reason_counts(self) -> dict:
        return {REASON_NAMES[i]: int((self.reason == i).sum()) for i in range(len(REASON_NAMES)) if (self.reason == i).any()}

    def summary(self) -> dict:
        return dict(n=int(len(self.admit)), admitted=int(self.admit.sum()), verified=self.n_verified, audit=self.n_audit,
                    deferred=self.n_deferred, exec_failed=self.n_failed, reasons=self.reason_counts())


class AdmissionGate:
    """Learning admission gate. `budget` is a fraction of the chunk if it is a float (0.3 = 30 %) or a number of executions if it is an int."""

    def __init__(self, mode: str = "strict", evidence: Optional[EvidenceModel] = None, budget=0.3, audit_rate: float = 0.05,
                 triage_low: float = 0.2, require_label_agreement: bool = True, relabel_from_evidence: bool = False,
                 seed: int = 0, check_invariant: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "none" and (evidence is None or not evidence.fitted):
            raise ValueError("selective and strict modes need a fitted EvidenceModel (fail-safe: no evidence, no admission)")
        if isinstance(budget, float) and not (0.0 < budget <= 1.0):
            raise ValueError("a float budget is a fraction of the chunk and must be in (0, 1]")
        if isinstance(budget, int) and budget < 0:
            raise ValueError("an int budget is a number of executions and must be >= 0")
        self.mode, self.evidence, self.budget = mode, evidence, budget
        self.audit_rate, self.triage_low, self.check_invariant = float(audit_rate), float(triage_low), bool(check_invariant)
        self.rng = np.random.default_rng(seed)
        strict = mode == "strict"
        self.policy = AdmitAllPolicy() if mode == "none" else EvidenceConsistencyPolicy(
            require_label_agreement, relabel_from_evidence, "deny" if strict else "admit")

    def budget_for(self, n: int) -> int:
        if isinstance(self.budget, float):
            return int(min(n, max(1, round(self.budget * n)))) if n else 0
        return int(min(n, self.budget))

    def calibrate_triage(self, p_attack_benign_calibration, quantile: float = 0.90) -> float:
        """Selective mode: set the fast-path threshold from the model's attack scores on benign CALIBRATION data."""
        p = np.asarray(p_attack_benign_calibration, float)
        self.triage_low = float(np.clip(np.quantile(p, quantile), 0.02, 0.5))
        return self.triage_low

    def decide(self, y_obs, verify: Optional[VerifyFn] = None, p_attack=None) -> GateResult:
        """y_obs: observed labels (0 = benign, >0 attack classes). verify(idx) -> (F, ok): evidence counts (len(idx), 7) and a bool
        mask of executions that produced telemetry, for the observations at positions idx of this chunk. p_attack: current model's
        P(attack) per observation (needed in selective mode)."""
        y_obs = np.asarray(y_obs).astype(int)
        n = len(y_obs)
        if n == 0:
            e = np.zeros(0, np.int8)
            return GateResult(np.zeros(0, bool), y_obs.copy(), e, e, e)
        if self.mode == "none":
            d = self.policy.decide(y_obs, np.zeros(n, np.int8), np.zeros(n, np.int8))
            return GateResult(d.admit, d.label, d.reason, np.full(n, -1, np.int8), np.full(n, V_UNAVAILABLE, np.int8))
        if verify is None:
            raise ValueError("selective and strict modes need a `verify` function that supplies the evidence")
        strict = self.mode == "strict"
        if strict:
            need, fast = np.arange(n), np.array([], int)
        else:
            if p_attack is None:
                raise ValueError("selective mode needs p_attack (the current model's attack scores) for triage")
            p_attack = np.asarray(p_attack, float)
            mask = (p_attack >= self.triage_low) | (y_obs != 0)
            need, fast = np.flatnonzero(mask), np.flatnonzero(~mask)
        need = need.copy()
        self.rng.shuffle(need)
        budget = self.budget_for(n)
        audit = np.array([], int)
        if not strict and self.audit_rate > 0 and len(fast):
            k = min(int(round(self.audit_rate * len(fast))), budget)
            if k > 0:                                     # audit slots are reserved out of the budget
                audit = self.rng.choice(fast, size=k, replace=False)
        take = need[: max(0, budget - len(audit))]
        deferred = need[len(take):]
        vidx = np.concatenate([take, audit]).astype(int)
        route = np.full(n, ROUTE_FAST, np.int8)
        route[vidx] = ROUTE_VERIFIED
        route[deferred] = ROUTE_DEFERRED
        verdict = np.full(n, V_UNAVAILABLE, np.int8)
        n_failed = 0
        if len(vidx):
            F, ok = verify(vidx)
            F, ok = np.asarray(F, float), np.asarray(ok, bool)
            if F.shape != (len(vidx), len(EVENT_KINDS)) or ok.shape != (len(vidx),):
                raise ValueError(f"verify must return (F, ok) with F of shape ({len(vidx)}, {len(EVENT_KINDS)}) and ok of shape ({len(vidx)},)")
            n_failed = int((~ok).sum())
            verdict[vidx] = self.evidence.verdicts(F, ok).verdict
        d = self.policy.decide(y_obs, route, verdict)
        if self.check_invariant:
            bad = check_gate_invariant(self.policy, y_obs, route, verdict, d)
            if bad:
                raise AssertionError(f"gate contract violated for {bad} observations (bug)")
        return GateResult(d.admit, d.label, d.reason, route, verdict, len(vidx), len(audit), len(deferred), n_failed)

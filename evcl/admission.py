"""evcl.admission -- Learning Admission Gate.

Policy semantics (EvidenceConsistencyPolicy):
  * verified path : admit iff evidence is decisive AND consistent with the observed label
                    (or relabelled from evidence if configured).
  * fast path     : UNVERIFIED traffic is admitted only if fastpath == "admit". In that case A(x)=1 does NOT
                    imply that behavioural evidence exists for x; the literal predicate "A(x)=1 => Evidence(x)
                    satisfies the policy" holds only with fastpath="deny" or triage mode "all".
  * deferred      : verification budget exhausted -> not admitted."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import V_BENIGN, V_MALICIOUS, V_UNAVAILABLE, V_UNCERTAIN

ROUTE_FAST, ROUTE_VERIFIED, ROUTE_DEFERRED = 0, 1, 2
(R_FAST_ADMIT, R_FAST_DENY, R_VERIFIED_OK, R_MISMATCH, R_UNCERTAIN, R_UNAVAILABLE, R_DEFERRED,
 R_ADMIT_ALL, R_RELABELLED, R_KNN_PASS, R_KNN_REJECT) = range(11)
REASON_NAMES = ["fast_admit", "fast_deny", "verified_consistent", "verdict_label_mismatch", "uncertain_evidence",
                "unavailable_evidence", "budget_deferred", "admit_all", "relabelled_from_evidence",
                "knn_pass", "knn_reject"]


@dataclass
class AdmissionDecision:
    admit: np.ndarray
    label: np.ndarray
    reason: np.ndarray


class AdmissionPolicy:
    def decide(self, y_obs, route, verdict) -> AdmissionDecision:
        raise NotImplementedError


class AdmitAllPolicy(AdmissionPolicy):
    def decide(self, y_obs, route, verdict):
        n = len(y_obs)
        return AdmissionDecision(np.ones(n, bool), y_obs.copy(), np.full(n, R_ADMIT_ALL, np.int8))


class EvidenceConsistencyPolicy(AdmissionPolicy):
    def __init__(self, require_label_agreement=True, relabel_from_evidence=False, fastpath="admit"):
        if fastpath not in ("admit", "deny"):
            raise ValueError(fastpath)
        self.agree, self.relabel, self.fastpath = require_label_agreement, relabel_from_evidence, fastpath

    def decide(self, y_obs, route, verdict):
        n = len(y_obs)
        admit = np.zeros(n, bool)
        label = y_obs.copy()
        reason = np.full(n, R_FAST_DENY, np.int8)
        fast = route == ROUTE_FAST
        if self.fastpath == "admit":
            admit[fast] = True
            reason[fast] = R_FAST_ADMIT
        reason[route == ROUTE_DEFERRED] = R_DEFERRED
        ver = route == ROUTE_VERIFIED
        ben, mal = ver & (verdict == V_BENIGN), ver & (verdict == V_MALICIOUS)
        reason[ver & (verdict == V_UNCERTAIN)] = R_UNCERTAIN
        reason[ver & (verdict == V_UNAVAILABLE)] = R_UNAVAILABLE
        if self.agree:
            ok = (ben & (y_obs == 0)) | (mal & (y_obs > 0))
            admit[ok] = True
            reason[ok] = R_VERIFIED_OK
            if self.relabel:
                fix = ben & (y_obs > 0)
                admit[fix], label[fix], reason[fix] = True, 0, R_RELABELLED
            mism = (ben & (y_obs > 0)) | (mal & (y_obs == 0))
            reason[mism & ~admit] = R_MISMATCH
        else:
            dec = ben | mal
            admit[dec] = True
            reason[dec] = R_VERIFIED_OK
        return AdmissionDecision(admit, label, reason)


def check_gate_invariant(policy, y_obs, route, verdict, d: AdmissionDecision) -> int:
    """Count violations of the implemented gate contract (implementation test, NOT a security proof)."""
    bad = 0
    a = d.admit
    if isinstance(policy, EvidenceConsistencyPolicy):
        bad += int(np.sum(a & (route == ROUTE_DEFERRED)))
        if policy.fastpath == "deny":
            bad += int(np.sum(a & (route == ROUTE_FAST)))
        ver = a & (route == ROUTE_VERIFIED)
        bad += int(np.sum(ver & ~np.isin(verdict, (V_BENIGN, V_MALICIOUS))))
        if policy.agree and not policy.relabel:
            bad += int(np.sum(ver & (verdict == V_BENIGN) & (d.label != 0)))
            bad += int(np.sum(ver & (verdict == V_MALICIOUS) & (d.label == 0)))
        if policy.agree and policy.relabel:
            bad += int(np.sum(ver & (verdict == V_MALICIOUS) & (d.label == 0)))
    return bad
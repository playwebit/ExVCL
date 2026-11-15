"""evcl.pipelines -- B0 frozen/reject-all, B1 accept-all, BR random-throttle control, B2 kNN-sanitiser (DEV PROXY),
B2b trimmed-loss filter (DEV PROXY), B3 execution-verified (modes none|selective|strict).
Pipelines receive ONLY ObservedChunk / DefenderView (no ground truth, no evaluation data, no class count).

B3 verification modes (cfg.verification.mode):
  none      : admit everything; no sandbox use (must reproduce B1 exactly -- tested).
  selective : verify AMBIGUOUS observations only (not confident-normal, or model/label disagreement);
              unverified fast-path traffic is admitted; audit slots are RESERVED out of the budget.
  strict    : verify EVERYTHING; nothing is admitted without decisive, label-consistent evidence;
              observations beyond the verification budget are rejected (deferred => not learned).
Fail-safe: if evidence is unavailable (execution failure, outage, unfitted fusion) the verdict is 'unavailable'
and the observation is NOT admitted on the verified route."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .admission import (R_ADMIT_ALL, R_KNN_PASS, R_KNN_REJECT, ROUTE_DEFERRED, ROUTE_FAST, ROUTE_VERIFIED,
                        AdmitAllPolicy, EvidenceConsistencyPolicy)
from .continual import KNNSanitizer, make_learner
from .core import EnvironmentNotSupported, RunConfig, make_rng
from .data import DefenderView, ObservedChunk, OracleChannel
from .evidence import (CalibratedFusion, EvidenceFuser, RuleBasedFusion, V_UNAVAILABLE, calibrate_thresholds)
from .sandbox import SandboxAdapter, SandboxRequest, build_simulated_sandbox
from .telemetry import normalize_events
from .triage import ModelTriage

perf = time.perf_counter


class MalformedInput(ValueError):
    """Observed chunk violates the input contract (shape / finiteness / label type)."""


def validate_observed(ch: ObservedChunk, n_features: int):
    X, y = np.asarray(ch.X), np.asarray(ch.y_obs)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise MalformedInput(f"expected X of shape (n,{n_features}), got {X.shape}")
    if len(y) != len(X) or len(ch.obs_id) != len(X):
        raise MalformedInput("X / y_obs / obs_id length mismatch")
    if not np.isfinite(X).all():
        raise MalformedInput("non-finite feature values")
    if y.size and not np.issubdtype(y.dtype, np.integer):
        raise MalformedInput("labels must be integers")


@dataclass
class ChunkOutcome:
    admit: np.ndarray
    label: np.ndarray
    reason: np.ndarray
    route: np.ndarray                     # 0 fast(unverified), 1 verified, 2 deferred, -1 not applicable
    n_verified: int = 0
    n_audit: int = 0
    n_deferred: int = 0
    n_exec_failed: int = 0
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class Calibration:
    """Explicit trusted labelled executions (for real adapters). The experimenter MUST attest that the calibration
    payloads are disjoint from every payload that appears in the evaluated stream."""
    requests: List[SandboxRequest]
    labels: np.ndarray                    # 1 = malicious
    attest_disjoint_from_stream: bool = False


def sim_calibration(n: int):
    """LEGACY (v1) label-built corpus; used only by legacy_config() for regression."""
    labels = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    reqs = [SandboxRequest(-1 - i, f"calib{i}") for i in range(len(labels))]
    oracle = OracleChannel([r.obs_id for r in reqs], labels, np.zeros(len(labels), bool))
    return Calibration(reqs, labels, attest_disjoint_from_stream=True), oracle


def _empty_outcome(n):
    return ChunkOutcome(np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.int8), np.full(n, -1, np.int8))


class BasePipeline:
    name = "base"
    telemetry_source = "NONE"
    triage_backend = "NONE"
    needs_oracle = False

    def __init__(self, cfg: RunConfig, view: DefenderView, seed: int):
        self.cfg, self.seed, self.n_features = cfg, seed, view.n_features
        self.rng = make_rng(seed, f"pipeline:{self.name}")
        self.learner = make_learner(cfg, view.n_features, seed)

    def setup(self, view: DefenderView):
        self.learner.bootstrap(view.bootstrap_X, view.bootstrap_y)

    def provenance(self) -> Dict:
        return {}

    def process_chunk(self, ch: ObservedChunk, t: int, oracle: Optional[OracleChannel] = None) -> ChunkOutcome:
        raise NotImplementedError


class B0Frozen(BasePipeline):
    """Trivial reject-all baseline: never learns from the stream (model frozen after bootstrap)."""
    name = "B0"

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        return _empty_outcome(len(ch))


class B1Naive(BasePipeline):
    name = "B1"

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        t0 = perf()
        pol = AdmitAllPolicy().decide(ch.y_obs, np.zeros(n, np.int8), np.zeros(n, np.int8))
        t1 = perf()
        self.learner.update(ch.X, ch.y_obs, ch.obs_id)
        return ChunkOutcome(pol.admit, pol.label, pol.reason, np.full(n, -1, np.int8),
                            timings=dict(admission=t1 - t0, learn=perf() - t1))


class RandomThrottle(BasePipeline):
    """CONTROL (not a defence): admits each observation with probability p, ignoring content."""
    name = "BR"

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        t0 = perf()
        keep = self.rng.random(n) < self.cfg.random_admit_p
        t1 = perf()
        self.learner.update(ch.X[keep], ch.y_obs[keep], ch.obs_id[keep])
        return ChunkOutcome(keep, ch.y_obs.copy(), np.where(keep, R_ADMIT_ALL, R_KNN_REJECT).astype(np.int8),
                            np.full(n, -1, np.int8), timings=dict(admission=t1 - t0, learn=perf() - t1))


class ClassBalancedThrottle(BasePipeline):
    """CONTROL (not a defence): admits a fixed fraction of each chunk with an EQUAL quota per observed label, ignoring content.
    Isolates the effect of pure data curation (class balance, less data) from any behavioural evidence."""
    name = "BC"

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        t0 = perf()
        keep = np.zeros(n, bool)
        labels = np.unique(ch.y_obs)
        if len(labels):
            q = int(round(self.cfg.curation_frac * n / len(labels)))
            for lab in labels:
                idx = np.flatnonzero(ch.y_obs == lab)
                keep[self.rng.choice(idx, size=min(len(idx), q), replace=False)] = True
        t1 = perf()
        self.learner.update(ch.X[keep], ch.y_obs[keep], ch.obs_id[keep])
        return ChunkOutcome(keep, ch.y_obs.copy(), np.where(keep, R_ADMIT_ALL, R_KNN_REJECT).astype(np.int8),
                            np.full(n, -1, np.int8), timings=dict(admission=t1 - t0, learn=perf() - t1))


class B2Robust(BasePipeline):
    """DEVELOPMENT PROXY (kNN label sanitiser). NOT a reproduction of a published defence; not SOTA.
    Look-ahead note: neighbours include other observations of the SAME chunk (transductive within-chunk)."""
    name = "B2"

    def __init__(self, cfg, view, seed):
        super().__init__(cfg, view, seed)
        self.san = KNNSanitizer(cfg.b2.k, cfg.b2.min_agreement)

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        if n == 0:
            return _empty_outcome(0)
        t0 = perf()
        bX, by, _ = self.learner.buffer.all()
        ref_X = np.vstack([self.learner.boot_X, bX])
        ref_y = np.concatenate([self.learner.boot_y, by])
        keep = self.san.keep_mask(self.learner.transform(ch.X), ch.y_obs, ref_X, ref_y)
        t1 = perf()
        self.learner.update(ch.X[keep], ch.y_obs[keep], ch.obs_id[keep])
        return ChunkOutcome(keep, ch.y_obs.copy(), np.where(keep, R_KNN_PASS, R_KNN_REJECT).astype(np.int8),
                            np.full(n, -1, np.int8), timings=dict(admission=t1 - t0, learn=perf() - t1))


class B2bTrimLoss(BasePipeline):
    """DEVELOPMENT PROXY: reject the `trim_frac` highest-loss samples (under the current model, observed labels)
    of each chunk -- a generic trimmed-loss filter, NOT a reproduction of any published method. Samples whose
    label the model does not know yet have undefined loss and are admitted (cannot be judged)."""
    name = "B2b"

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        if n == 0:
            return _empty_outcome(0)
        t0 = perf()
        loss = self.learner.loss_per_sample(ch.X, ch.y_obs)
        keep = np.ones(n, bool)
        known = ~np.isnan(loss)
        if known.sum() > 0 and self.cfg.trim_frac > 0:
            thr = np.quantile(loss[known], 1.0 - self.cfg.trim_frac)
            keep[known & (loss > thr)] = False
        t1 = perf()
        self.learner.update(ch.X[keep], ch.y_obs[keep], ch.obs_id[keep])
        return ChunkOutcome(keep, ch.y_obs.copy(), np.where(keep, R_KNN_PASS, R_KNN_REJECT).astype(np.int8),
                            np.full(n, -1, np.int8), timings=dict(admission=t1 - t0, learn=perf() - t1))


class B3Proposed(BasePipeline):
    name = "B3"

    def __init__(self, cfg, view, seed, sandbox: Optional[SandboxAdapter] = None,
                 calibration: Optional[Calibration] = None):
        super().__init__(cfg, view, seed)
        self.mode = cfg.verification.mode
        self.sandbox, self.explicit_calibration = None, calibration
        self.info: Dict = {"calibration_status": "n/a", "triage_low": float("nan"), "t_benign": float("nan"),
                           "t_malicious": float("nan"), "threshold_status": "n/a"}
        if self.mode == "none":
            self.policy = AdmitAllPolicy()
            return
        sc, ad = cfg.sandbox, cfg.admission
        if sandbox is None:
            if sc.backend != "simulated":
                raise EnvironmentNotSupported("backend != 'simulated': pass an explicit SandboxAdapter instance.")
            sandbox = build_simulated_sandbox(sc, seed)
        if sandbox.is_simulated and not cfg.allow_simulated_sandbox:
            raise EnvironmentNotSupported("Simulated sandbox requested but allow_simulated_sandbox=False.")
        if calibration is not None and not calibration.attest_disjoint_from_stream:
            raise ValueError("Calibration.attest_disjoint_from_stream must be True: calibration payloads must be "
                             "disjoint from stream payloads (otherwise train/test contamination of the fusion).")
        if calibration is None and not sandbox.is_simulated and cfg.calibration_source == "legacy_sim":
            raise ValueError("Real sandbox adapters require a calibration period or an explicit Calibration corpus.")
        self.sandbox = sandbox
        self.needs_oracle = bool(sandbox.is_simulated)
        self.telemetry_source = sandbox.source
        self.triage_backend = "USERSPACE_MODEL"
        strict = self.mode == "strict"
        self.triage = ModelTriage(cfg.triage.low_threshold, "all" if strict else "model")
        self.audit_rate = 0.0 if strict else ad.audit_rate
        self.fusion = CalibratedFusion(seed) if cfg.evidence.fusion == "calibrated" else RuleBasedFusion()
        self.fuser = EvidenceFuser(self.fusion, cfg.evidence.t_benign, cfg.evidence.t_malicious)
        self.fusion_ready = False
        self.policy = EvidenceConsistencyPolicy(ad.require_label_agreement, ad.relabel_from_evidence,
                                                "deny" if strict else "admit")

    def provenance(self):
        return dict(self.info, sandbox_model=self.cfg.sandbox.model if self.sandbox is not None else "n/a",
                    sandbox_control=self.cfg.sandbox.control if self.sandbox is not None else "n/a")

    def setup(self, view):
        super().setup(view)
        if self.mode == "none":
            return
        cfg, sb = self.cfg, self.sandbox
        if self.explicit_calibration is not None:
            cal = self.explicit_calibration
            reqs, labels, benign_X = cal.requests, np.asarray(cal.labels).astype(int), None
        elif cfg.calibration_source == "legacy_sim":
            cal, oracle = sim_calibration(cfg.sandbox.calib_n)
            reqs, labels, benign_X = cal.requests, cal.labels, None
            if sb.is_simulated:
                sb.set_oracle(oracle)
        else:                                          # calibration PERIOD: trusted labelled observations before the stream
            reqs = [SandboxRequest(-1 - i, f"calib{i}") for i in range(len(view.calib_y))]
            labels = (view.calib_y > 0).astype(int)
            benign_X = view.calib_X[view.calib_y == 0]
            if sb.is_simulated:
                sb.set_oracle(OracleChannel([r.obs_id for r in reqs], view.calib_y, np.zeros(len(reqs), bool)))
        res = sb.execute_many(reqs)
        ok = np.array([r.status == "ok" for r in res])
        self.info["calibration_ok_frac"] = float(ok.mean()) if len(ok) else 0.0
        if ok.sum() >= 20 and len(np.unique(labels[ok])) == 2:
            F = np.vstack([normalize_events(r.events) for r in res])
            self.fusion.fit(F[ok], labels[ok])
            self.fusion_ready = True
            self.info["calibration_status"] = "fitted"
            if cfg.evidence.threshold_source == "calibration":
                tb, tm, st = calibrate_thresholds(self.fusion.predict_proba(F[ok]), labels[ok] == 1,
                                                  cfg.evidence.target_fpr, cfg.evidence.target_fnr,
                                                  (cfg.evidence.t_benign, cfg.evidence.t_malicious))
                self.fuser.tb, self.fuser.tm = tb, tm
                self.info["threshold_status"] = st
            else:
                self.info["threshold_status"] = "fixed"
            self.info.update(t_benign=self.fuser.tb, t_malicious=self.fuser.tm)
        else:
            self.info["calibration_status"] = "FAILED: too few successful calibration executions -> evidence unavailable (fail-safe)"
        if cfg.triage.threshold_source == "calibration" and benign_X is not None and len(benign_X):
            p = self.learner.p_attack(benign_X)
            self.triage.low = float(np.clip(np.quantile(p, cfg.triage.benign_pass_quantile), 0.02, 0.5))
        self.info["triage_low"] = self.triage.low
        if hasattr(sb, "activate_control"):
            sb.activate_control()                      # negative controls take effect at deployment
        sb.n_executed = 0

    def _prioritise(self, need, ch, tri):
        """Order the observations that need verification; the first `budget` get a sandbox slot (the rest are deferred).
        random (default; identical RNG use to the frozen v2 code) | balanced (round-robin over observed labels) |
        uncertainty (largest model/label disagreement first) | novelty_balanced (rare-in-memory labels first)."""
        mode = self.cfg.sandbox.allocation
        if mode == "random" or len(need) == 0:
            self.rng.shuffle(need)
            return need
        y = ch.y_obs[need]
        if mode == "balanced":
            groups = [self.rng.permutation(need[y == lab]) for lab in np.unique(y)]
            order = []
            for k in range(max(len(g) for g in groups)):
                for g in groups:
                    if k < len(g):
                        order.append(g[k])
            return np.asarray(order, dtype=need.dtype)
        if mode == "uncertainty":
            disagree = np.abs(tri.score[need] - (y > 0).astype(float)) + 1e-9 * self.rng.random(len(need))
            return need[np.argsort(-disagree)]
        # novelty_balanced: weighted random order without replacement, weight = 1 / (1 + count of that label in replay memory)
        _, by, _ = self.learner.buffer.all()
        counts = np.bincount(by.astype(int), minlength=int(max(y.max(), 0)) + 1) if len(by) else np.zeros(int(y.max()) + 1, int)
        w = 1.0 / (1.0 + counts[np.clip(y, 0, len(counts) - 1)])
        keys = self.rng.random(len(need)) ** (1.0 / w)
        return need[np.argsort(-keys)]

    def process_chunk(self, ch, t, oracle=None):
        validate_observed(ch, self.n_features)
        n = len(ch)
        if n == 0:
            return _empty_outcome(0)
        if self.mode == "none":
            t0 = perf()
            d = self.policy.decide(ch.y_obs, np.zeros(n, np.int8), np.zeros(n, np.int8))
            t1 = perf()
            self.learner.update(ch.X, ch.y_obs, ch.obs_id)
            return ChunkOutcome(d.admit, d.label, d.reason, np.full(n, -1, np.int8),
                                timings=dict(admission=t1 - t0, learn=perf() - t1))
        tm = {}
        if self.sandbox.is_simulated:
            if oracle is None:
                raise ValueError("simulated sandbox needs the runner's OracleChannel")
            self.sandbox.set_oracle(oracle)
        t0 = perf()
        tri = self.triage.route(ch.X, ch.y_obs, self.learner)       # scores use the model BEFORE this chunk's update
        tm["triage"] = perf() - t0
        budget = self.cfg.sandbox.budget_per_chunk
        need = self._prioritise(np.flatnonzero(tri.verify_needed), ch, tri)
        fast_idx = np.flatnonzero(~tri.verify_needed)
        audit = np.array([], int)
        if self.audit_rate > 0 and len(fast_idx):
            k = min(int(round(self.audit_rate * len(fast_idx))), budget)
            if k > 0:                                   # audit slots are RESERVED out of the budget, so a saturated
                audit = self.rng.choice(fast_idx, size=k, replace=False)   # queue can never silently disable audits
        take = need[: max(0, budget - len(audit))]
        deferred = need[len(take):]
        vidx = np.concatenate([take, audit]).astype(int)
        route = np.full(n, ROUTE_FAST, np.int8)
        route[vidx] = ROUTE_VERIFIED
        route[deferred] = ROUTE_DEFERRED
        reqs = [SandboxRequest(int(ch.obs_id[i]), str(int(ch.obs_id[i]))) for i in vidx]
        res = self.sandbox.execute_many(reqs)
        tm["sandbox"] = sum(r.wall_seconds for r in res)
        t2 = perf()
        verdict = np.full(n, V_UNAVAILABLE, np.int8)
        n_failed = 0
        if len(res):
            ok = np.array([r.status == "ok" for r in res])
            n_failed = int((~ok).sum())
            if self.fusion_ready:
                F = np.vstack([normalize_events(r.events) for r in res])
                verdict[vidx] = self.fuser.fuse(F, ok).verdict
        tm["fusion"] = perf() - t2
        t3 = perf()
        d = self.policy.decide(ch.y_obs, route, verdict)
        tm["admission"] = perf() - t3
        t4 = perf()
        self.learner.update(ch.X[d.admit], d.label[d.admit], ch.obs_id[d.admit])
        tm["learn"] = perf() - t4
        return ChunkOutcome(d.admit, d.label, d.reason, route, n_verified=len(vidx), n_audit=len(audit),
                            n_deferred=len(deferred), n_exec_failed=n_failed, timings=tm)


def make_pipeline(name, cfg, view, seed, sandbox=None, calibration=None):
    table = {"B0": B0Frozen, "B1": B1Naive, "BR": RandomThrottle, "BC": ClassBalancedThrottle, "B2": B2Robust, "B2b": B2bTrimLoss}
    if name in table:
        return table[name](cfg, view, seed)
    if name == "B3":
        return B3Proposed(cfg, view, seed, sandbox, calibration)
    raise ValueError(name)
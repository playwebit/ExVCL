"""evcl.phase2 -- Phase 2: the EVCL admission gate on RECORDED host telemetry (two real VMs, real services, lab traffic).

What is real here: every observation is one recorded lab session; its behavioural evidence is the strace-derived event
counts measured on the target VM while the session ran; its network features come from the tcpdump capture of the same session.
What is still simulated: the POISONING (label flips and a feature trigger are applied to the recorded sessions by the attack lab).
The recorded evidence of a poisoned observation is always that of the TRUE underlying session (a backdoor copy carries the evidence
of the session it was copied from), so the evidence never reads the label the learner sees.

Information barriers: the recorded adapter is built by the experiment RUNNER from the attacked stream (obs_id -> session); the
defender only sends an opaque payload reference and receives events. Calibration uses the separate PILOT recording.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import experiment as E
from .audit import assert_results_valid, banner
from .checkpoint import CheckpointManager, SeedRegistry
from .core import (DEFAULT_ATTACKS, DataCfg, RunConfig, apply_overrides, cfg_to_dict, config_hash, make_rng,
                   validate_config)
from .data import CHUNK_ID_STRIDE, Chunk, StreamData, assert_disjoint_rows, chunk_counts
from .evaluation import add_pi
from .pipelines import Calibration
from .poisoning import Attack, LabelPoisoning, _clone, _inject, _window
from .realstudy import LEARNERS, _sha
from .sandbox import (GlobalRandomEvidenceSandbox, SandboxAdapter, SandboxRequest, SandboxResult,
                      ShuffledEvidenceSandbox, UnavailableSandbox)
from .telemetry import EVENT_KINDS, TelemetryEvent

# ----------------------------------------------------------------------------- classes and features
CLASS_ID = {"benign_web": 0, "benign_ssh": 0, "benign_ftp": 0, "ssh_bruteforce": 1, "ftp_bruteforce": 2, "port_scan": 3,
            # hard-case block: the label follows the generator's intent and maps onto the SAME learner classes
            "hard_benign_typos": 0, "hard_benign_admin": 0, "hard_slow_ssh_bf": 1, "hard_mimic_ssh_bf": 1,
            "hard_slow_ftp_bf": 2, "hard_mimic_ftp_bf": 2}
CLASS_NAMES = ["BENIGN", "SSH-bruteforce", "FTP-bruteforce", "port-scan"]
# network features per session (from the pcap). net_frac_to_target is NOT used: it depends on capture direction and was unstable.
NET_FEATURES = ("n_pkts", "n_bytes", "duration_s", "pkts_per_s", "bytes_per_s", "mean_len", "std_len", "n_src_ports", "n_dst_ports",
                "n_flows", "syn", "rst", "fin", "psh", "mean_iat", "std_iat")
XCOLS = tuple("x_" + f for f in NET_FEATURES)
EVCOLS = tuple("ev_" + k for k in EVENT_KINDS)
PHASE2_ATTACKS = ("backdoor", "label_poisoning")
NULLS = (("shuffled-evidence", "within-batch shuffled"), ("global-random-evidence", "global-random (calibration pool)"))
MIX_Q = (0.25, 0.5)            # share of observations that keep their REAL evidence in the dose-response variants (0 = global-random, 1 = real)


@dataclasses.dataclass(frozen=True)
class P2Layout:
    n_chunks: int = 12
    chunk_size: int = 90
    benign_frac: float = 0.7
    novel_start: int = 6                 # the last class (FTP brute force, or port scan if include_scan) first appears at this chunk
    include_scan: bool = False           # DECISION: port scans are host-invisible (see protocol); False keeps them out of the gate study
    final_frac: float = 0.2              # latest share of each class's EVALUATION sessions held out as the final test period
    bootstrap_per_class: int = 40        # trusted bootstrap sessions per base class (benign x2), taken from the PILOT
    dev_per_class: int = 30              # development-test sessions per class, from the PILOT (latest of each class)
    budget_frac: float = 0.30            # verification budget per chunk as a share of the chunk size
    sweep_fracs: tuple = (0.6, 1.0)
    collector_failure_utc: str = "2026-10-27T06:11:10Z"   # web sessions after this time without traced evidence are excluded
    trigger_features: int = 3
    trigger_sigma: float = 4.0
    backdoor_budget: float = 0.2
    label_budget: float = 0.2
    attack_start: int = 3
    attack_end: int = 10


# ----------------------------------------------------------------------------- session tables
def prepare_sessions(obs: pd.DataFrame, tag: str) -> pd.DataFrame:
    """observations.csv (kit build_dataset output, with pcap features) -> the table the study uses."""
    need = {"session_id", "cls", "malicious", "start_ms", "end_ms", "covered"} | set(EVCOLS) | {"net_" + f for f in NET_FEATURES}
    miss = need - set(obs.columns)
    if miss:
        raise ValueError(f"{tag}: observations table lacks columns {sorted(miss)}")
    o = obs.copy().reset_index(drop=True)
    o["tag"] = tag
    o["ref"] = tag + ":" + o.session_id.astype(str)
    if o.ref.duplicated().any():
        raise ValueError(f"{tag}: duplicate session ids")
    o["stratum"] = np.where(o.cls.astype(str).str.startswith("hard_"), "hard", "templated")
    o["cid"] = o.cls.map(CLASS_ID).fillna(-1).astype(int)      # idle sessions: -1 (no network features, used for calibration/attribution only)
    o["covered"] = o.covered.astype(bool)
    for f in NET_FEATURES:
        v = o["net_" + f].to_numpy(float)
        o["x_" + f] = np.sign(v) * np.log1p(np.abs(v))
    o["slot_ms"] = o.start_ms.astype(float)
    if "rc" in o.columns:
        o["rc"] = o.rc.fillna(0).astype(int)
    return o


def apply_generator_failure_rule(df: pd.DataFrame, tag: str):
    """Pre-specified rule, applied to EVERY recording and class: a session whose generator did not complete (return code != 0, so the
    intended action did not happen and its label no longer describes what was recorded) is excluded. The rule uses the generator's return
    code only; it never looks at evidence, features, labels' outcomes or gate decisions."""
    if "rc" not in df.columns:
        return df, dict(tag=tag, rc_available=False, n_sessions=int(len(df)), n_excluded=0, excluded_by_class={})
    bad = df[df.rc != 0]
    info = dict(tag=tag, rc_available=True, n_sessions=int(len(df)), n_excluded=int(len(bad)),
                excluded_by_class={str(k): int(v) for k, v in bad.cls.value_counts().items()},
                kept_by_class={str(k): int(v) for k, v in df.drop(index=bad.index).cls.value_counts().items()})
    return df.drop(index=bad.index).reset_index(drop=True), info


def apply_collector_exclusion(main: pd.DataFrame, topup: Optional[pd.DataFrame], cutoff_utc: str):
    """Pre-specified rule: benign-web sessions at or after `cutoff_utc` that have NO traced evidence are excluded as a collector
    failure (nginx restarted and was never re-attached). Each excluded session is replaced by the next top-up web session (slotting:
    the replacement takes the excluded session's chronological position). Uses only coverage, never class results."""
    cutoff_ms = pd.Timestamp(cutoff_utc).value / 1e6
    web = main[main.cls == "benign_web"]
    bad = web[(web.start_ms >= cutoff_ms) & (~web.covered)].sort_values("start_ms")
    rep = pd.DataFrame()
    if topup is not None and len(topup):
        rep = topup[(topup.cls == "benign_web") & topup.covered].sort_values("start_ms")
    n_rep = min(len(bad), len(rep))
    out = main.drop(index=bad.index).copy()
    if n_rep:
        r = rep.iloc[:n_rep].copy()
        r["slot_ms"] = bad.start_ms.to_numpy(float)[:n_rep]
        out = pd.concat([out, r], ignore_index=True)
    info = dict(n_web_main=int(len(web)), n_excluded=int(len(bad)), n_topup_usable=int(len(rep)), n_replaced=int(n_rep),
                n_excluded_unreplaced=int(len(bad) - n_rep), n_topup_unused=int(max(0, len(rep) - n_rep)),
                topup_covered_frac=float(topup[topup.cls == "benign_web"].covered.mean()) if topup is not None and len(topup) else float("nan"),
                cutoff_utc=cutoff_utc)
    return out, info


def events_of(row) -> List[TelemetryEvent]:
    return [TelemetryEvent(k, "recorded", int(row["ev_" + k])) for k in EVENT_KINDS if int(row["ev_" + k]) > 0]


def events_by_ref(*tables: pd.DataFrame) -> Dict[str, List[TelemetryEvent]]:
    out: Dict[str, List[TelemetryEvent]] = {}
    for t in tables:
        for ref, evc in zip(t.ref, t[list(EVCOLS)].to_numpy()):
            out[ref] = [TelemetryEvent(k, "recorded", int(c)) for k, c in zip(EVENT_KINDS, evc) if c > 0]
    return out


# ----------------------------------------------------------------------------- stream
def n_classes_for(L: P2Layout) -> int:
    return 4 if L.include_scan else 3


def base_classes_for(L: P2Layout):
    return tuple(range(n_classes_for(L) - 1))


def schedule_for(L: P2Layout):
    K = n_classes_for(L)
    return tuple((c, 0, L.n_chunks) for c in range(K - 1)) + ((K - 1, L.novel_start, L.n_chunks),)


def restrict_classes(df: pd.DataFrame, L: P2Layout) -> pd.DataFrame:
    """Without include_scan, port-scan sessions are removed from the study tables (stream, calibration, tests); their evidence is
    reported separately as a descriptive finding."""
    return df if L.include_scan else df[df.cls != "port_scan"].reset_index(drop=True)


def make_p2_cfg(L: P2Layout, csv_path: str, seeds, run_mode: str, learner: str = "L2", name: str = "evcl_phase2",
                attacks: Optional[dict] = None) -> RunConfig:
    data = DataCfg(kind="csv", n_features=len(NET_FEATURES), n_classes=n_classes_for(L), n_chunks=L.n_chunks, chunk_size=L.chunk_size,
                   benign_frac=L.benign_frac, bootstrap_per_class=L.bootstrap_per_class, test_per_class=L.dev_per_class,
                   calib_per_class=20, final_test_per_class=1, schedule=schedule_for(L), base_classes=base_classes_for(L),
                   trigger_features=L.trigger_features, trigger_sigma=L.trigger_sigma, csv_path=csv_path, csv_label_col="cls",
                   csv_benign_label="benign", csv_time_col="start_ms", csv_final_frac=L.final_frac)
    atk = attacks or {"backdoor": dict(budget=L.backdoor_budget, start=L.attack_start, end=L.attack_end),
                      "label_poisoning": dict(budget=L.label_budget, start=L.attack_start, end=L.attack_end)}
    cfg = RunConfig(experiment_name=name, run_mode=run_mode, seeds=tuple(int(s) for s in seeds), data=data, attacks=atk,
                    sandbox=dataclasses.replace(RunConfig().sandbox, backend="external",
                                                budget_per_chunk=int(round(L.budget_frac * L.chunk_size))))
    return apply_overrides(cfg, LEARNERS[learner]) if LEARNERS[learner] else cfg


def _dedupe(cal: pd.DataFrame, ev: pd.DataFrame, hard: Optional[pd.DataFrame] = None):
    """Drop sessions whose feature row duplicates an earlier one (calibration recording first). Idle sessions are not part of the stream."""
    parts = [cal[cal.cid >= 0].assign(role="cal"), ev[ev.cid >= 0].assign(role="eval")]
    if hard is not None:
        parts.append(hard[hard.cid >= 0].assign(role="hard"))
    allr = pd.concat(parts, ignore_index=True)
    dup = allr.duplicated(subset=list(XCOLS), keep="first")
    keep = allr[~dup]
    return (keep[keep.role == "cal"].copy(), keep[keep.role == "eval"].copy(),
            keep[keep.role == "hard"].copy() if hard is not None else None, int(dup.sum()))


def build_phase2_stream(cal: pd.DataFrame, ev: pd.DataFrame, cfg: RunConfig, seed: int, include_final: bool,
                        hard: Optional[pd.DataFrame] = None) -> StreamData:
    """Default: stream and final test come from the evaluation recording `ev`. With `hard` given (hard-only stream): the STREAM is the
    hard-case recording, while bootstrap/calibration/dev stay on the pilot and the final test stays on the templated evaluation recording."""
    d = cfg.data
    schedule = d.schedule
    K = d.n_classes
    cal_k, ev_k, hard_k, n_dup = _dedupe(cal, ev, hard)
    if len(cal_k) and len(ev_k) and not float(cal.end_ms.max()) < float(ev.start_ms.min()):
        raise ValueError("calibration recording overlaps the evaluation recording in time (must be strictly earlier)")
    if hard is not None and not float(cal.end_ms.max()) < float(hard.start_ms.min()):
        raise ValueError("calibration recording overlaps the hard-case recording in time (must be strictly earlier)")
    rng = make_rng(seed, "data")
    xc = list(XCOLS)
    demand = {c: 0 for c in range(K)}
    for t in range(d.n_chunks):
        for c, n in chunk_counts(d, schedule, t).items():
            demand[c] += n
    boot, dev, calib_rows, pools, pool_refs, fin_parts = [], [], [], {}, {}, []
    for c in range(K):
        cc = cal_k[cal_k.cid == c].sort_values("start_ms")
        n_b = d.bootstrap_per_class * (2 if c == 0 else 1) if c in d.base_classes else 0
        n_dev = min(d.test_per_class, max(1, len(cc) - n_b - 1))
        if len(cc) < n_b + n_dev:
            raise ValueError(f"class {CLASS_NAMES[c]}: only {len(cc)} calibration sessions after de-duplication, need {n_b}+{n_dev}")
        if n_b:
            boot.append((c, cc.iloc[:n_b][xc].to_numpy(float)))
        dev.append((c, cc.iloc[len(cc) - n_dev:][xc].to_numpy(float)))
        if c in d.base_classes:
            mid = cc.iloc[n_b:len(cc) - n_dev]
            if len(mid):
                calib_rows.append((c, mid[xc].to_numpy(float)))
        ee = ev_k[ev_k.cid == c].sort_values("slot_ms")
        n_fin = int(np.ceil(d.csv_final_frac * len(ee)))
        stream_part, fin_part = ee.iloc[:len(ee) - n_fin], ee.iloc[len(ee) - n_fin:]
        if hard_k is not None:
            stream_part = hard_k[hard_k.cid == c]
        if len(stream_part) < demand[c]:
            raise ValueError(f"class {CLASS_NAMES[c]}: {len(stream_part)} stream sessions available but the layout needs {demand[c]}; "
                             "reduce n_chunks/chunk_size or the novel-class window")
        perm = rng.permutation(len(stream_part))             # seed-dependent chunk order over the fixed recorded sessions
        sp = stream_part.iloc[perm]
        pools[c] = sp[xc].to_numpy(float)
        pool_refs[c] = sp.ref.tolist()
        fin_parts.append((c, fin_part[xc].to_numpy(float)))
    if not calib_rows:
        raise ValueError("no calibration-period rows left for the defender view")
    cursor = {c: 0 for c in range(K)}
    chunks, uid, uid_ref = [], 0, {}
    for t in range(d.n_chunks):
        counts = chunk_counts(d, schedule, t)
        Xs, ys, rs = [], [], []
        for c in sorted(counts):
            i, n = cursor[c], counts[c]
            cursor[c] = i + n
            Xs.append(pools[c][i:i + n]); ys.append(np.full(n, c, np.int64)); rs.extend(pool_refs[c][i:i + n])
        X, y = np.vstack(Xs), np.concatenate(ys)
        p = rng.permutation(len(y))
        X, y = X[p], y[p]
        rs = [rs[j] for j in p]
        n = len(y)
        u = np.arange(uid, uid + n, dtype=np.int64)
        for k, r in zip(u, rs):
            uid_ref[int(k)] = r
        uid += n
        chunks.append(Chunk(X=X, y_obs=y.copy(), y_true=y.copy(), is_poison=np.zeros(n, bool), kind=np.zeros(n, np.int8),
                            evasive=np.zeros(n, bool), trigger=np.zeros(n, bool), uid=u, evasion=np.zeros(n, np.int8)))

    def cat(parts):
        return np.vstack([x for _, x in parts]), np.concatenate([np.full(len(x), c, np.int64) for c, x in parts])
    bX, by = cat(boot)
    cX, cy = cat(calib_rows)
    dev_set = cat(dev)
    eval_sets = {"dev_pre": dev_set, "dev_post": dev_set}
    if include_final:
        eval_sets["final"] = cat(fin_parts)
    stream_rows = np.vstack([c.X for c in chunks])
    trusted = np.vstack([bX, cX])
    for name, (eX, _) in eval_sets.items():
        assert_disjoint_rows(eX, trusted, f"eval set {name} and trusted bootstrap/calibration data")
        assert_disjoint_rows(eX, stream_rows, f"eval set {name} and stream chunks")
    if include_final:
        assert_disjoint_rows(eval_sets["final"][0], dev_set[0], "final and development test sets")
    assert_disjoint_rows(bX, cX, "bootstrap and calibration periods")
    assert_disjoint_rows(trusted, stream_rows, "bootstrap/calibration and stream chunks")
    dcount = bX.shape[1]
    mu, sd = bX[by == 0].mean(0), bX[by == 0].std(0) + 1e-9
    trig_idx = np.sort(rng.choice(dcount, size=min(d.trigger_features, dcount), replace=False))
    first, last = {}, {}
    for c, s_, e in schedule:
        first[c] = min(first.get(c, 10 ** 9), s_)
        last[c] = max(last.get(c, -1), e)
    meta = dict(dataset="recorded_sessions_hard" if hard is not None else "recorded_sessions", is_synthetic=False,
                split_type="recorded_sessions_calibration_then_evaluation",
                temporal_split=True, entity_split=False, entity_purged_rows=0, dedup_dropped=n_dup, overlap_check="passed",
                feature_mean=mu, feature_std=sd, next_uid=uid, uid_ref=uid_ref, schedule=tuple(schedule), class_first_chunk=first,
                class_last_chunk=last, base_classes=tuple(d.base_classes), label_noise_rate=0.0, drift_chunk=-1,
                has_final=include_final, calib_classes=sorted(set(int(c) for c in np.unique(cy))),
                eval_trigger=dict(idx=trig_idx.tolist(), vals=(mu[trig_idx] + d.trigger_sigma * sd[trig_idx]).tolist()),
                class_centroids={c: pools[c][:300].mean(0) for c in range(1, K)}, class_names=CLASS_NAMES[:K])
    return StreamData(bX, by, cX, cy, chunks, eval_sets, K, dcount, meta).freeze()


# ----------------------------------------------------------------------------- attacks with recorded-evidence provenance
class RecordedBackdoor(Attack):
    """Backdoor: copies of attack observations carrying a fixed feature trigger, labelled benign. Each copy keeps the SESSION it was
    copied from, so its recorded evidence is the evidence of a real malicious session (the trigger changes features, not behaviour)."""
    name = "backdoor"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        trig = out.meta["eval_trigger"]
        idx_f, vals = np.asarray(trig["idx"]), np.asarray(trig["vals"])
        for t in range(s, e):
            ch = out.chunks[t]
            cand = np.flatnonzero((ch.y_true > 0) & (~ch.is_poison))
            n = int(round(self.params["budget"] * len(ch)))
            if len(cand) == 0 or n == 0:
                continue
            pick = rng.choice(cand, size=n, replace=True)
            X = ch.X[pick] + rng.normal(size=(n, ch.X.shape[1])) * 0.05 * out.meta["feature_std"]
            X[:, idx_f] = vals
            src = [out.meta["uid_ref"][int(ch.uid[i])] for i in pick]
            u0 = out.meta["next_uid"]
            _inject(out, t, X, np.zeros(n, np.int64), ch.y_true[pick], "backdoor", rng, trigger=np.ones(n, bool))
            for k, r in enumerate(src):
                out.meta["uid_ref"][u0 + k] = r
        return out


def build_scenario_p2(cfg: RunConfig, cal, ev, seed: int, attack: str, hard=None) -> StreamData:
    base = build_phase2_stream(cal, ev, cfg, seed, include_final=(cfg.run_mode == "final"), hard=hard)
    if attack == "none":
        return base
    if attack == "backdoor":
        return RecordedBackdoor(**cfg.attacks["backdoor"]).apply(base, make_rng(seed, "attack:backdoor"))
    if attack == "label_poisoning":
        return LabelPoisoning(**cfg.attacks["label_poisoning"]).apply(base, make_rng(seed, "attack:label_poisoning"))
    raise ValueError(f"attack {attack!r} has no recorded-evidence implementation in Phase 2")


# ----------------------------------------------------------------------------- recorded adapters and controls
class RecordedSessionSandbox(SandboxAdapter):
    """Replays the recorded evidence of the session behind each payload reference. `table` maps payload_ref -> events; stream
    observations use str(obs_id) (built by the runner from the attacked stream), calibration executions use the pilot session ref."""
    mode = "RECORDED"
    source = "RECORDED"
    is_simulated = False

    def __init__(self, table: Dict[str, List[TelemetryEvent]]):
        self.table = table

    def execute(self, req: SandboxRequest) -> SandboxResult:
        t0 = time.perf_counter()
        ev = self.table.get(req.payload_ref)
        if ev is None:
            return SandboxResult(req.obs_id, "error", [], 0.0, self.mode, False, "no recording for payload_ref")
        return SandboxResult(req.obs_id, "ok", list(ev), time.perf_counter() - t0, self.mode, False)


class MixedEvidenceSandbox(GlobalRandomEvidenceSandbox):
    """Dose-response control: after activation each observation keeps its REAL recorded evidence with probability q and otherwise
    receives evidence drawn at random from the calibration pool. q = 1 is real evidence, q = 0 is the global-random null."""

    def __init__(self, inner, rng, q: float, pool_cap=4000):
        super().__init__(inner, rng, pool_cap)
        self.q = float(q)

    def execute_many(self, reqs):
        res = self.inner.execute_many(reqs)
        self.n_executed += len(reqs)
        oks = [i for i, r in enumerate(res) if r.status == "ok"]
        if not self.active:
            for i in oks:
                if len(self.pool) < self.pool_cap:
                    self.pool.append(list(res[i].events))
            return res
        for i in oks:
            if self.rng.random() >= self.q and self.pool:
                ev = self.pool[int(self.rng.integers(0, len(self.pool)))]
                res[i] = SandboxResult(res[i].obs_id, "ok", list(ev), res[i].wall_seconds, res[i].mode, False)
        return res


class SupportModel:
    """Open-set check on evidence: evidence is FAMILIAR if it lies within the support of the known-benign OR the known-malicious calibration
    sessions (mean distance to the k nearest references, in standardised log-count space, below the q-quantile of the same distance among
    the references themselves). Evidence resembling neither is UNFAMILIAR: absence of familiar evidence is not evidence of benign behaviour."""

    def __init__(self, k: int = 3, q: float = 0.99):
        self.k, self.q = k, q
        self.thr = {0: np.inf, 1: np.inf}
        self.ref = {0: None, 1: None}

    def fit(self, F: np.ndarray, y: np.ndarray):
        Z = np.log1p(np.asarray(F, float))
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Z = (Z - self.mu) / self.sd
        for c in (0, 1):
            R = Z[np.asarray(y) == c]
            if len(R) > self.k + 1:
                D = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
                np.fill_diagonal(D, np.inf)
                self.thr[c] = float(np.quantile(np.sort(D, 1)[:, :self.k].mean(1), self.q))
                self.ref[c] = R
        return self

    def unfamiliar(self, F: np.ndarray) -> np.ndarray:
        Z = (np.log1p(np.asarray(F, float)) - self.mu) / self.sd
        ok = np.zeros(len(Z), bool)
        for c in (0, 1):
            R = self.ref[c]
            if R is None:
                continue
            D = np.sqrt(((Z[:, None, :] - R[None, :, :]) ** 2).sum(-1))
            ok |= np.sort(D, 1)[:, :self.k].mean(1) <= self.thr[c] + 1e-9
        return ~ok


class SupportCheckSandbox(SandboxAdapter):
    """Modification M1. Wraps an evidence adapter; during calibration it collects the evidence of the calibration sessions (labels come from the
    trusted calibration corpus), after activation it reports UNFAMILIAR evidence as unavailable, so the gate defers it instead of reading it as benign."""
    mode = "RECORDED+SUPPORT"
    source = "RECORDED"
    is_simulated = False

    def __init__(self, inner, label_of: Dict[str, int], k: int = 3, q: float = 0.99):
        self.inner, self.label_of, self.k, self.q = inner, label_of, k, q
        self.active = False
        self.F, self.y = [], []
        self.model = None
        self.n_executed = 0
        self.n_unfamiliar = 0

    def execute(self, req):
        return self.execute_many([req])[0]

    @staticmethod
    def _vec(events):
        v = np.zeros(len(EVENT_KINDS))
        for e in events:
            v[EVENT_KINDS.index(e.kind)] += e.count
        return v

    def execute_many(self, reqs):
        res = self.inner.execute_many(reqs)
        self.n_executed += len(reqs)
        if not self.active:
            for q, r in zip(reqs, res):
                if r.status == "ok" and q.payload_ref in self.label_of:
                    self.F.append(self._vec(r.events)); self.y.append(int(self.label_of[q.payload_ref]))
            return res
        out = []
        for r in res:
            if r.status == "ok" and self.model.unfamiliar(self._vec(r.events)[None, :])[0]:
                self.n_unfamiliar += 1
                out.append(SandboxResult(r.obs_id, "error", [], r.wall_seconds, r.mode, False, "unfamiliar evidence"))
            else:
                out.append(r)
        return out

    def activate_control(self):
        self.model = SupportModel(self.k, self.q).fit(np.vstack(self.F), np.array(self.y))
        self.active = True


def recorded_table(stream: StreamData, ev_events: Dict[str, List[TelemetryEvent]], cal_refs: Sequence[str]):
    table = {}
    uid_ref = stream.meta["uid_ref"]
    for t, ch in enumerate(stream.chunks):
        ids = ch.obs_ids(t)
        for pos, u in enumerate(ch.uid):
            table[str(int(ids[pos]))] = ev_events[uid_ref[int(u)]]
    for r in cal_refs:
        table[r] = ev_events[r]
    return table


def make_calibration(cal: pd.DataFrame) -> Calibration:
    """Pilot sessions (ALL classes, idle included) as the trusted labelled calibration corpus; disjoint in time from every evaluation session."""
    reqs = [SandboxRequest(-1 - i, ref) for i, ref in enumerate(cal.ref)]
    return Calibration(reqs, cal.malicious.to_numpy(int), attest_disjoint_from_stream=True)


def make_sandbox(vname: str, seed: int, table, label_of: Optional[Dict[str, int]] = None):
    inner = RecordedSessionSandbox(table)
    tag = vname.split("|")[1] if "|" in vname else ""
    if tag.endswith("support-check"):                      # "support-check" (M1) and "m2+support-check" (M1 on the M2 calibration)
        return SupportCheckSandbox(inner, label_of or {})
    if tag == "shuffled-evidence":
        return ShuffledEvidenceSandbox(inner, make_rng(seed, "shuffle"))
    if tag == "global-random-evidence":
        return GlobalRandomEvidenceSandbox(inner, make_rng(seed, "global_random"))
    if tag.startswith("mix-"):
        return MixedEvidenceSandbox(inner, make_rng(seed, "mix" + tag), float(tag[4:]))
    if tag == "sandbox-outage":
        return UnavailableSandbox(inner)
    return inner


def p2_variants(L: P2Layout, sweep: bool = True, modifications: bool = False) -> Dict[str, Dict]:
    v = {"NONE": {"verification.mode": "none"}}
    for m, b in (("SELECTIVE", E.SEL), ("STRICT", E.STR)):
        v[m] = dict(b)
        if modifications:                                  # M1 support check, M2 calibration enriched with the development recording, both
            v[f"{m}|support-check"] = dict(b)
            v[f"{m}|m2"] = dict(b)
            v[f"{m}|m2+support-check"] = dict(b)
        v[f"{m}|shuffled-evidence"] = dict(b)
        v[f"{m}|global-random-evidence"] = dict(b)
        for q in MIX_Q:
            v[f"{m}|mix-{q}"] = dict(b)
        v[f"{m}|sandbox-outage"] = dict(b)
        if sweep:
            for f in L.sweep_fracs:
                v[f"{m}|budget-{int(round(f * L.chunk_size))}"] = {**b, "sandbox.budget_per_chunk": int(round(f * L.chunk_size))}
    return v


def hard_layout(L: P2Layout) -> P2Layout:
    """Hard-only stream: the 400 hard sessions are the whole stream (benign 200, SSH-type attacks 100, FTP-type attacks 100)."""
    return dataclasses.replace(L, n_chunks=6, chunk_size=40, benign_frac=0.5, novel_start=3, attack_start=1, attack_end=5)


def evidence_error_table(cal: pd.DataFrame, tables: Sequence[pd.DataFrame], L: P2Layout, seed: int = 0) -> pd.DataFrame:
    """Per-session evidence verdicts of a gate calibrated ONLY on the pilot (same fusion and threshold rule as the pipeline).
    Shows where the evidence itself is right or wrong, independent of the learner. Port scans (if present) are reported as is."""
    from .evidence import CalibratedFusion, EvidenceFuser, calibrate_thresholds
    c = restrict_classes(cal, L)
    F, y = c[list(EVCOLS)].to_numpy(float), c.malicious.to_numpy(int)
    fus = CalibratedFusion(seed).fit(F, y)
    ev_cfg = RunConfig().evidence
    tb, tm, status = calibrate_thresholds(fus.predict_proba(F), y == 1, ev_cfg.target_fpr, ev_cfg.target_fnr, (ev_cfg.t_benign, ev_cfg.t_malicious))
    fu = EvidenceFuser(fus, tb, tm)
    rows = []
    for t in tables:
        if t is None or not len(t):
            continue
        Ft = t[list(EVCOLS)].to_numpy(float)
        b = fu.fuse(Ft, np.ones(len(Ft), bool))
        d = t[["cls", "stratum", "malicious"]].copy()
        d["p_malicious"], d["verdict"] = b.p_malicious, b.verdict
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    g = d.groupby(["stratum", "cls", "malicious"])
    out = g.agg(n=("verdict", "size"), mean_p=("p_malicious", "mean"), p_min=("p_malicious", "min"), p_max=("p_malicious", "max")).reset_index()
    for k, nm in ((0, "frac_verdict_benign"), (1, "frac_verdict_malicious"), (2, "frac_verdict_uncertain")):
        out[nm] = g.verdict.apply(lambda v, k=k: float((v == k).mean())).to_numpy()
    out["frac_correct"] = np.where(out.malicious == 1, out.frac_verdict_malicious, out.frac_verdict_benign)
    out["threshold_status"] = status
    return out


# ----------------------------------------------------------------------------- hypotheses (written into the frozen protocol)
HYPOTHESES_PHASE2 = [
    "PHASE 2 (recorded host telemetry from real services on two lab VMs). Question: does RECORDED behavioural evidence let the strict admission gate admit a smaller share of poison than uninformative evidence?",
    "Data: pilot recording (calibration and development only) and evaluation recording (stream and final test), strictly later in time; benign-web sessions after the collector failure without traced evidence are excluded and replaced by the top-up web sessions in their chronological slots.",
    "Poisoning is simulated on the recorded sessions (label flips attack->benign; a feature trigger on copies of attack sessions labelled benign). The recorded evidence of a poisoned observation is always the evidence of its TRUE underlying session.",
    "Primary outcome (lower is better): admitted-poison share = poisoned observations / all admitted observations in the attacked run (column admitted_poison_frac), seed-level means, attacks backdoor and label_poisoning, mode STRICT.",
    "Primary tests: STRICT with real recorded evidence versus each of 2 nulls (within-batch shuffled, global-random from the calibration pool) on each of the 2 attacks = 4 paired tests, exact two-sided Wilcoxon, bootstrap CI, Holm over the 4.",
    "CONFIRMED only if (a) real evidence has the lower admitted-poison share than BOTH nulls with Holm p < 0.05 and CI above 0 on BOTH attacks; (b) the share is non-increasing across evidence mixes q = 0 (global-random), 0.25, 0.5, 1.0 (real) (tolerance 0.005 on the means) on both attacks; (c) NO confounding: real evidence admits at least as much clean data as each null (clean-run accept fraction not lower by more than 0.005).",
    "If (a) and (b) hold but (c) fails: CONFOUNDED; if (a) holds but (b) fails: PARTIAL; if (a) fails: NOT CONFIRMED. Attack effectiveness on accept-all (backdoor ASR rise >= 0.20, label-poisoning macro-F1 drop >= 0.03) is reported descriptively and does not change the verdict.",
    "SELECTIVE is analysed with the same tests as a secondary, descriptive analysis. Also descriptive: sandbox-outage control, operating-point sweep (verification budget 60% and 100% of the chunk), matched evidence-free curation (BC/BR) at the admitted fraction of the real SELECTIVE/STRICT, sanitisers B2/B2b, downstream macro-F1, evidence coverage and collector overhead.",
    "Scope: the sessions are templated lab sessions of 3 service classes plus a port scan; separability of the recorded evidence is expected to be very high. A confirmation therefore validates the recorded-evidence channel; it does not test ambiguous or adaptive behaviour and does not establish resistance to adaptive attackers.",
    "Seeds are new and used only here; negative results will be reported as they are.",
]


# ----------------------------------------------------------------------------- study runner
HYPOTHESES_PHASE2_HARD = [
    "PHASE 2, HARD-CASE STRATUM (pre-specified secondary analysis, frozen together with the primary study before any trial). Same gate, learner, nulls and statistics as the primary study; the gate is calibrated ONLY on the templated pilot recording and never sees a hard session before it is evaluated.",
    "Stream: the 400 hard-case sessions (slow brute force and mimicry brute force over SSH and FTP; benign logins with 1-3 mistyped passwords; benign admin logins that read /etc/passwd). Labels follow the generator's intent and map onto the learner's existing classes (BENIGN, SSH-bruteforce, FTP-bruteforce). Bootstrap, calibration and development data come from the pilot; the final test period is the held-out part of the templated evaluation recording.",
    "Same primary outcome and the same 4 paired tests as the primary study (STRICT, real recorded evidence versus within-batch-shuffled and global-random evidence, backdoor and label_poisoning, Holm over 4), and the same verdict rule. The verdict is reported for this stratum as a SECONDARY result: a failure here does not change the verdict of the primary study.",
    "Per-session evidence table (descriptive): for every hard session type, the share of sessions whose fused evidence is called malicious, benign or uncertain by a gate calibrated on the pilot only, with the share correct against the generator's intent.",
    "Expectation written before the data: slow and mimicry brute force and benign sessions with mistyped passwords produce overlapping host evidence, so the evidence is expected to be less accurate here than on templated sessions; errors are reported as they are and are not tuned away.",
    "Scope: sessions with a VALID password (a stolen-credential attacker) produce host evidence identical to a legitimate login and are out of scope for host telemetry; no adaptive attacker is modelled; port scans are not in this stratum.",
    "Seeds are new and used only here; negative results will be reported as they are.",
]


def _prepared_csv(path, cal, ev, hard=None, dev=None):
    cols = ["ref", "tag", "cls", "malicious", "start_ms", "end_ms", "covered", "slot_ms", *XCOLS, *EVCOLS]
    pd.concat([t[cols] for t in (cal, ev, hard, dev) if t is not None], ignore_index=True).to_csv(path, index=False)
    return path


def run_phase2_study(root: str, cal: pd.DataFrame, ev: pd.DataFrame, L: P2Layout, seeds, run_mode: str = "development",
                     stage_name: str = "phase2", attacks=PHASE2_ATTACKS, sweep: bool = True, extra_pipelines=("B2", "B2b"),
                     learner: str = "L2", notes: str = "", exclusion_info: Optional[dict] = None, verbose: bool = True,
                     hard: Optional[pd.DataFrame] = None, freeze_only: bool = False, hypotheses: Optional[list] = None,
                     dev: Optional[pd.DataFrame] = None, variants_only: Optional[Sequence[str]] = None):
    """hard=None: primary study. hard=<hard-case table>: hard-only stream (use hard_layout(L)). freeze_only=True writes/checks the frozen
    protocol and returns without running a trial (so that BOTH protocols can be frozen before either study runs)."""
    cal, ev = restrict_classes(cal, L), restrict_classes(ev, L)
    dev = restrict_classes(dev, L) if dev is not None else None
    HYP = hypotheses or (HYPOTHESES_PHASE2 if hard is None else HYPOTHESES_PHASE2_HARD)
    stage_dir = os.path.join(root, "runs", stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    csv_path = _prepared_csv(os.path.join(stage_dir, "phase2_sessions_prepared.csv"), cal, ev, hard, dev)
    cfg = make_p2_cfg(L, csv_path, seeds, run_mode, learner)
    for w in validate_config(cfg):
        if "labelling channel" not in w:
            print("WARNING:", w, flush=True)
    build_scenario_p2(cfg, cal, ev, int(seeds[0]), "none", hard)     # feasibility pre-check before anything is frozen
    variants = p2_variants(L, sweep, modifications=dev is not None)
    if variants_only is not None:
        variants = {k: v for k, v in variants.items() if k in set(variants_only)}
    design = dict(variants=sorted(variants), attacks=list(attacks), layout=dataclasses.asdict(L), nulls=[k for k, _ in NULLS],
                  mix_q=list(MIX_Q), extra_pipelines=list(extra_pipelines), learner=learner, dataset_sha256=_sha(csv_path),
                  n_cal_sessions=int(len(cal)), n_eval_sessions=int(len(ev)), n_hard_sessions=int(len(hard)) if hard is not None else 0, n_dev_sessions=int(len(dev)) if dev is not None else 0,
                  stratum="hard" if hard is not None else "templated", exclusion=exclusion_info or {})
    if run_mode == "final":
        pf = os.path.join(stage_dir, "protocol_frozen.json")
        if os.path.exists(pf):
            E.check_protocol(pf, cfg, design, cfg.seeds, HYP)
        else:
            E.freeze_protocol(pf, cfg, design, cfg.seeds, HYP)
    if freeze_only:
        print(f"protocol for stage {stage_name!r} frozen/checked; no trial run", flush=True)
        return None
    ev_events = events_by_ref(*(t for t in (cal, ev, hard, dev) if t is not None))
    calib = make_calibration(cal)
    label_of = dict(zip(cal.ref, cal.malicious.astype(int)))
    calib_m2, label_of_m2, cal_m2_refs = None, None, None
    if dev is not None:
        if not float(dev.start_ms.min()) > float(cal.end_ms.max()):
            raise ValueError("the development recording must follow the pilot")
        for nm, t in (("evaluation", ev), ("hard-case", hard)):
            if t is not None and not float(dev.end_ms.max()) < float(t.start_ms.min()):
                raise ValueError(f"the development recording must precede the {nm} recording")
        cal2 = pd.concat([cal, dev], ignore_index=True)
        calib_m2 = make_calibration(cal2)
        label_of_m2 = dict(zip(cal2.ref, cal2.malicious.astype(int)))
        cal_m2_refs = list(cal2.ref)
    ck = CheckpointManager(stage_dir, "evcl_phase2", config_hash(cfg))
    ck.write_json("config.json", cfg_to_dict(cfg))
    SeedRegistry(stage_dir, "evcl_phase2").register(cfg.run_mode, cfg.seeds, config_hash(cfg))
    vcfgs = {v: apply_overrides(cfg, ov) for v, ov in variants.items()}
    t0 = time.time()
    res: List[Dict] = []
    cal_refs = list(cal.ref)
    for seed in cfg.seeds:
        for attack in ("none",) + tuple(attacks):
            stream = build_scenario_p2(cfg, cal, ev, int(seed), attack, hard)
            table = recorded_table(stream, ev_events, cal_refs + (cal_m2_refs or []))
            for vname, vcfg in vcfgs.items():
                key = E.trial_key(vname, "B3", attack, seed)
                if ck.has_trial(key):
                    res.append(ck.load_trial(key))
                    continue
                m2 = "|m2" in vname
                res.append(E.run_trial(vcfg, "B3", stream, seed, attack, vname, ck,
                                       sandbox=make_sandbox(vname, int(seed), table, label_of_m2 if m2 else label_of),
                                       calibration=calib_m2 if m2 else calib))
            for p in extra_pipelines:
                key = E.trial_key("base", p, attack, seed)
                res.append(ck.load_trial(key) if ck.has_trial(key) else E.run_trial(cfg, p, stream, seed, attack, "base", ck))
            for rv in [v for v in variants if v in ("SELECTIVE", "STRICT") or v.split("|")[-1] in ("support-check", "m2", "m2+support-check")]:   # matched evidence-free curation at the admitted fraction of each gate
                row = next((r for r in res if r["variant"] == rv and r["pipeline"] == "B3" and r["attack"] == attack and r["seed"] == seed), None)
                if row is None:
                    continue
                f = float(np.clip(row["accept_frac"], 0.02, 1.0))
                for p, ov in (("BC", {"curation_frac": f}), ("BR", {"random_admit_p": f})):
                    vname = f"matched-{rv}"
                    key = E.trial_key(vname, p, attack, seed)
                    if ck.has_trial(key):
                        r = ck.load_trial(key)
                    else:
                        r = E.run_trial(apply_overrides(cfg, ov), p, stream, seed, attack, vname, ck)
                    r["matched_fraction"] = f
                    r["config_hash_cell"] = r["config_hash"]
                    r["config_hash"] = config_hash(cfg)
                    res.append(r)
        if verbose:
            print(f"[seed {seed}] done | {len(res)} trials | elapsed {time.time() - t0:.0f}s", flush=True)
    df = add_pi(E.results_to_df(res))
    assert_results_valid(df)
    df["dose"] = "phase2"
    df.to_csv(os.path.join(stage_dir, "phase2_results_all.csv"), index=False)
    json.dump(dict(design=design, hypotheses=HYP, run_mode=run_mode, seeds=list(map(int, cfg.seeds)), notes=notes,
                   config_hash=config_hash(cfg)), open(os.path.join(stage_dir, "phase2_design.json"), "w"), indent=2, default=str)
    print(f"PHASE 2 STUDY done: {len(df)} trials in {time.time() - t0:.0f}s | {banner(df)}", flush=True)
    return df
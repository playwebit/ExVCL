"""exvcl.selftest -- tests of the library (fast, no network, no root) and, optionally, of the bundled research engine."""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from typing import Callable, Dict

import numpy as np

from . import (AdmissionGate, ChunkVerifier, EvidenceModel, ExVCL, EVENT_KINDS, Verdict, CalibrationError, counts_from_strace_text,
               make_learner)
from .checks import run_gate_check, synthetic_sessions
from .gate import GateResult


def _cal(seed=0, n=60):
    F, y = synthetic_sessions(n, seed)
    return F, y


def t_evidence_fit_and_failsafe():
    F, y = _cal()
    ev = EvidenceModel().fit(F, y)
    v = ev.verdicts(F).verdict
    assert (v == y).mean() > 0.9, "verdicts should agree with labels on separable calibration data"
    try:
        EvidenceModel().verdicts(F)
    except CalibrationError:
        pass
    else:
        raise AssertionError("an unfitted model must refuse to produce verdicts")
    try:
        EvidenceModel().fit(F[:8], y[:8])
    except CalibrationError:
        pass
    else:
        raise AssertionError("too few calibration executions must raise CalibrationError")
    try:
        EvidenceModel().fit(F[y == 0], y[y == 0])
    except CalibrationError:
        pass
    else:
        raise AssertionError("one-sided calibration must raise CalibrationError")
    ok = np.ones(len(F), bool); ok[:5] = False
    assert (ev.verdicts(F, ok).verdict[:5] == Verdict.UNAVAILABLE).all(), "failed executions are UNAVAILABLE"


def t_evidence_roundtrip():
    F, y = _cal(1)
    ev = EvidenceModel(support_check=True).fit(F, y)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ev.json"); ev.save(p); ev2 = EvidenceModel.load(p)
    Q = np.vstack([F[:30], F[:5] * 40 + 500])
    a, b = ev.verdicts(Q), ev2.verdicts(Q)
    assert np.array_equal(a.verdict, b.verdict) and np.allclose(a.p_malicious, b.p_malicious, equal_nan=True), "save/load must not change verdicts"


def t_evidence_drop_nuisance_feature():
    F, y = _cal(8)
    F = F.copy(); F[:, 6] = 595 + 2 * y            # a nuisance count that happens to correlate with the label in calibration
    ev = EvidenceModel(drop=("syscall_total",)).fit(F, y)
    Q = F.copy(); Q[:, 6] = 594                      # same behaviour, different nuisance value
    assert (ev.verdicts(Q).verdict == ev.verdicts(F).verdict).all(), "a dropped count must not influence verdicts"
    try:
        EvidenceModel(drop=("nope",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown drop name must raise")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.json"); ev.save(p); assert EvidenceModel.load(p).drop == ("syscall_total",)


def t_support_check_defers_unfamiliar():
    F, y = _cal(2)
    ev = EvidenceModel(support_check=True).fit(F, y)
    far = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]), (6, 1))        # a "quiet" execution unlike any calibration session
    assert (ev.verdicts(far).verdict == Verdict.UNAVAILABLE).all(), "unfamiliar evidence must be unavailable, not benign"
    assert (EvidenceModel().fit(F, y).verdicts(far).verdict != Verdict.UNAVAILABLE).all(), "without M1 it is read as a verdict"


def t_gate_none():
    g = AdmissionGate("none")
    r = g.decide(np.array([0, 1, 0, 1]))
    assert r.admit.all() and (r.label == np.array([0, 1, 0, 1])).all()


def _chunk(seed=0, n=40):
    F, y = _cal(seed, 40)
    ev = EvidenceModel().fit(F[:60], y[:60])
    return ev, F[60:60 + n], y[60:60 + n]


def t_gate_strict_contract():
    ev, F, y = _chunk()
    y_obs = y.copy(); flip = (y == 1)[:]; flip[::2] = False; y_obs[flip] = 0           # label-flip some attacks to benign
    g = AdmissionGate("strict", ev, budget=1.0)
    r = g.decide(y_obs, ChunkVerifier(F))
    assert not r.admit[flip & (r.verdict == Verdict.MALICIOUS)].any(), "label/evidence mismatch must be rejected"
    assert r.admit[~flip].mean() > 0.8, "consistent observations are admitted"
    assert not r.admit[r.route == 2].any() and not r.admit[r.verdict >= 2].any()
    g2 = AdmissionGate("strict", ev, budget=5)
    r2 = g2.decide(y_obs, ChunkVerifier(F))
    assert r2.n_verified == 5 and r2.n_deferred == len(y) - 5 and r2.admit.sum() <= 5, "budget caps verification; overflow is deferred, not learned"


def t_gate_selective_and_audit():
    ev, F, y = _chunk(3)
    p = np.where(y == 1, 0.9, 0.05)                                                        # confident model
    g = AdmissionGate("selective", ev, budget=1.0, audit_rate=0.2, triage_low=0.2)
    r = g.decide(y.copy(), ChunkVerifier(F), p)
    fast = (p < 0.2) & (y == 0)
    assert r.n_audit > 0 and ((r.route == 1) & fast).sum() == r.n_audit, "audit slots verify a share of the fast path"
    assert (r.admit[fast & (r.route == 0)]).all(), "unverified confident-normal traffic takes the fast path"
    try:
        g.decide(y.copy(), ChunkVerifier(F))
    except ValueError:
        pass
    else:
        raise AssertionError("selective mode needs p_attack")


def t_gate_failsafe_and_contract():
    ev, F, y = _chunk(4)
    g = AdmissionGate("strict", ev, budget=1.0)
    r = g.decide(y.copy(), lambda idx: (F[idx], np.zeros(len(idx), bool)))
    assert not r.admit.any() and r.n_failed == len(y), "failed executions mean no admission"
    try:
        g.decide(y.copy(), lambda idx: (F[idx][:, :3], np.ones(len(idx), bool)))
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong-shaped verify result must raise")
    try:
        AdmissionGate("strict", None)
    except ValueError:
        pass
    else:
        raise AssertionError("strict mode needs a fitted evidence model")


def t_chunk_verifier_controls():
    F, y = _cal(5)
    pool, Fs = F[:40], F[40:80]
    idx = np.arange(20)
    real, _ = ChunkVerifier(Fs)(idx)
    assert np.array_equal(real, Fs[idx])
    sh, _ = ChunkVerifier(Fs, control="shuffled", rng=np.random.default_rng(1))(idx)
    assert sorted(map(tuple, sh)) == sorted(map(tuple, Fs[idx])) and not np.array_equal(sh, Fs[idx]), "shuffled permutes within the batch"
    gr, _ = ChunkVerifier(Fs, control="global_random", pool=pool, rng=np.random.default_rng(2))(idx)
    assert all(tuple(r) in set(map(tuple, pool)) for r in gr), "global_random draws only from the calibration pool"
    m1, _ = ChunkVerifier(Fs, control="mix", pool=pool, q=1.0)(idx)
    m0, _ = ChunkVerifier(Fs, control="mix", pool=pool, q=0.0, rng=np.random.default_rng(3))(idx)
    assert np.array_equal(m1, Fs[idx]) and all(tuple(r) in set(map(tuple, pool)) for r in m0), "mix q=1 is real, q=0 is the null"


def t_learners():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)); y = (X[:, 0] > 0).astype(int); X[y == 1, :3] += 2
    for kind in ("softmax", "mlp", "forest"):
        ex = ExVCL(6, "none", learner=kind, seed=0)
        ex.bootstrap(X[:100], y[:100])
        for a in range(100, 200, 20):
            ex.process_chunk(X[a:a + 20], y[a:a + 20])
        assert (ex.predict(X) == y).mean() > 0.8, f"{kind} learner should learn the toy task"


def t_strace_parser():
    txt = ('100  execve("/usr/bin/sh", ["sh","-c","id"], 0x1 /* 1 vars */) = 0\n'
           '100  openat(AT_FDCWD, "/etc/shadow", O_RDONLY) = -1 EACCES (Permission denied)\n'
           '100  openat(AT_FDCWD, "/tmp/x", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 3\n'
           '100  connect(3, {sa_family=AF_INET, sin_port=htons(9), sin_addr=inet_addr("127.0.0.1")}, 16) = -1 ECONNREFUSED\n'
           '100  setuid(0) = -1 EPERM\n+++ exited with 0 +++\n')
    c = dict(zip(EVENT_KINDS, counts_from_strace_text(txt)))
    assert c["process_spawn"] == 1 and c["shell_exec"] == 1 and c["sensitive_file_access"] == 1 and c["file_write"] == 1
    assert c["net_connect_out"] == 1 and c["priv_change"] == 1 and c["syscall_total"] == 5, c


def t_end_to_end_attribution():
    F, y = synthetic_sessions(60, 7)
    rep = run_gate_check(F, y, seed=7, repeats=2)
    v = rep["variants"]
    assert v["none/real"]["poison_admitted"] > 0.99 and v["strict/real"]["poison_admitted"] < 0.2, v
    assert v["strict/global_random"]["poison_admitted"] > v["strict/real"]["poison_admitted"] + 0.15, "null evidence must admit more poison"


LIB_TESTS: Dict[str, Callable] = {n[2:]: f for n, f in sorted(globals().items()) if n.startswith("t_") and callable(f)}


def run_library() -> Dict[str, str]:
    out = {}
    for name, fn in LIB_TESTS.items():
        try:
            fn(); out[name] = "PASS"
        except Exception as e:                                                         # noqa: BLE001
            out[name] = "FAIL: " + "".join(traceback.format_exception_only(type(e), e)).strip()[:300]
    return out


def run_engine() -> Dict[str, str]:
    """The research engine's own 25 tests (information barriers, leakage, collectors' parsers, gate contract)."""
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collectors")
    if cdir not in sys.path:
        sys.path.insert(0, cdir)
    from evcl.phase2_selftest import run_all as r2
    from evcl.phase3_selftest import run_all as r3
    return {"engine/" + k: v for k, v in {**r2(), **r3()}.items()}


def run_all(engine: bool = True) -> Dict[str, str]:
    r = run_library()
    if engine:
        r.update(run_engine())
    return r

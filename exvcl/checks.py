"""exvcl.checks -- end-to-end checks of the library: a synthetic demo and a real-telemetry check for a VM.

The checks exercise the whole path: evidence counts -> calibrated evidence model -> admission gate -> continual learner, with
label-flip backdoor poison in the stream and the attribution controls. They test that the LIBRARY works as designed. They are not
research results: the network features are simulated and the stream is small.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from .api import ExVCL
from .attribution import ChunkVerifier, admitted_poison_share, clean_retention
from .evidence import EvidenceModel
from evcl.telemetry import EVENT_KINDS

D = 8                                     # simulated network-feature dimension
VARIANTS = (("none", "real"), ("strict", "real"), ("strict", "shuffled"), ("strict", "global_random"), ("selective", "real"))


def synthetic_sessions(n_per_class: int, seed: int):
    """Class-conditional evidence counts (benign / attack-like) with overlap, in the order of EVENT_KINDS."""
    rng = np.random.default_rng(seed)
    mb = np.array([3.0, 0.2, 6.0, 0.3, 1.5, 0.02, 400.0]); ma = np.array([11.0, 3.0, 10.0, 5.0, 12.0, 0.3, 900.0])
    Fb = rng.poisson(mb * rng.uniform(0.6, 1.5, (n_per_class, 1)))
    Fa = rng.poisson(ma * rng.uniform(0.6, 1.5, (n_per_class, 1)))
    F = np.vstack([Fb, Fa]).astype(float); y = np.r_[np.zeros(n_per_class), np.ones(n_per_class)].astype(int)
    p = rng.permutation(len(y))
    return F[p], y[p]


def _feats(y, rng, trigger=False):
    X = rng.normal(size=(len(y), D))
    X[y == 1, :4] += 2.5
    if trigger:
        X[:, D - 1] += 5.0
    return X


def run_gate_check(F, y, seed: int = 0, repeats: int = 5, chunk: int = 16, poison_share: float = 0.6, budget=1.0) -> Dict:
    """Split the sessions into calibration and stream, poison a share of the malicious stream sessions (label flipped to benign + trigger
    feature), run the gate variants and report admitted-poison share, clean retention and backdoor success on a held-out set."""
    F, y = np.asarray(F, float), np.asarray(y, int)
    rng0 = np.random.default_rng(seed)
    idx = rng0.permutation(len(y))
    cal, st = [], []
    for c in (0, 1):
        ids = idx[y[idx] == c]; h = len(ids) // 2
        cal += list(ids[:h]); st += list(ids[h:])
    cal, st = np.array(cal), np.array(st)
    ev = EvidenceModel(seed=seed).fit(F[cal], y[cal])
    v_st = ev.verdicts(F[st]).verdict
    ev_acc = float((v_st == y[st]).mean()); ev_unc = float(np.isin(v_st, (2, 3)).mean())
    rows: Dict[str, List[Dict]] = {f"{m}/{c}": [] for m, c in VARIANTS}
    for rep in range(repeats):
        rng = np.random.default_rng(10_000 + 97 * seed + rep)
        X_cal = _feats(y[cal], rng)
        order = rng.permutation(len(st)); S = st[order]
        yS = y[S]; X = _feats(yS, rng)
        mal = np.flatnonzero(yS == 1)
        pois = np.zeros(len(S), bool); pois[rng.choice(mal, int(round(poison_share * len(mal))), replace=False)] = True
        X[pois, D - 1] += 5.0
        y_obs = yS.copy(); y_obs[pois] = 0
        yt = np.r_[np.zeros(300), np.ones(300)].astype(int)
        X_clean, X_trig = _feats(yt, rng), _feats(np.ones(300, int), rng, trigger=True)
        for mode, control in VARIANTS:
            ex = ExVCL(D, mode, ev if mode != "none" else None, "softmax", budget=budget, seed=rep)
            ex.bootstrap(X_cal, y[cal])
            if mode == "selective":
                ex.calibrate_triage(X_cal[y[cal] == 0])
            vr = np.random.default_rng(555 + rep)
            admit = np.zeros(len(S), bool); nver = ndef = 0
            for a in range(0, len(S), chunk):
                sl = np.arange(a, min(a + chunk, len(S)))
                ver = ChunkVerifier(F[S[sl]], control=control, pool=F[cal], rng=vr)
                r = ex.process_chunk(X[sl], y_obs[sl], ver)
                admit[sl] = r.admit; nver += r.n_verified; ndef += r.n_deferred
            pred = ex.predict(np.vstack([X_clean, X_trig]))
            bal = 0.5 * ((pred[:300] == 0).mean() + (pred[300:600] == 1).mean())
            asr = float((ex.predict(X_trig) == 0).mean())
            rows[f"{mode}/{control}"].append(dict(poison_admitted=admitted_poison_share(admit, pois), clean_retained=clean_retention(admit, pois),
                                                  asr=asr, bal_acc_clean=float(bal), verified=nver, deferred=ndef))
    summary = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    return dict(n_sessions=int(len(y)), n_calibration=int(len(cal)), n_stream=int(len(st)), repeats=repeats, poison_share=poison_share,
                budget=budget, evidence=dict(accuracy=ev_acc, undecided=ev_unc, t_benign=ev.t_benign, t_malicious=ev.t_malicious,
                                             status=ev.status), variants=summary)


def evaluate(report: Dict, recording_ok: Optional[bool] = None) -> List[Dict]:
    """Pass/fail criteria. Thresholds are loose on purpose (small samples); they detect a broken pipeline, not small differences."""
    v, e = report["variants"], report["evidence"]
    sr, ss, sg, nn = v["strict/real"], v["strict/shuffled"], v["strict/global_random"], v["none/real"]
    out = []
    def add(name, ok, detail): out.append(dict(check=name, passed=bool(ok), detail=detail))
    if recording_ok is not None:
        add("real strace sessions recorded and parsed", recording_ok, "every session produced syscalls")
    add("evidence model separates the held-out sessions", e["accuracy"] >= 0.85, f"verdict accuracy {e['accuracy']:.2f} (need >= 0.85)")
    add("accept-all (none) admits all poison", nn["poison_admitted"] >= 0.99, f"{nn['poison_admitted']:.2f}")
    add("strict gate with real evidence admits little poison", sr["poison_admitted"] <= 0.20, f"{sr['poison_admitted']:.2f} (need <= 0.20)")
    add("strict gate keeps most clean data at full budget", sr["clean_retained"] >= 0.60, f"{sr['clean_retained']:.2f} (need >= 0.60)")
    add("protection comes from the evidence (global-random null admits more poison)",
        sg["poison_admitted"] >= sr["poison_admitted"] + 0.15, f"null {sg['poison_admitted']:.2f} vs real {sr['poison_admitted']:.2f}")
    add("shuffled-evidence null does not beat real evidence", ss["poison_admitted"] >= sr["poison_admitted"], f"shuffled {ss['poison_admitted']:.2f} vs real {sr['poison_admitted']:.2f}")
    if nn["asr"] >= 0.3:
        add("backdoor success is lower under the strict gate than under accept-all", sr["asr"] < nn["asr"], f"ASR {sr['asr']:.2f} vs {nn['asr']:.2f}")
    return out


def format_report(report: Dict, checks: List[Dict], extra: Optional[Dict] = None) -> str:
    L = []
    if extra:
        for k, val in extra.items():
            L.append(f"  {k}: {val}")
    e = report["evidence"]
    L.append(f"  sessions: {report['n_sessions']} ({report['n_calibration']} calibration, {report['n_stream']} stream); repeats {report['repeats']}; poison share of malicious stream {report['poison_share']}")
    L.append(f"  evidence model: held-out verdict accuracy {e['accuracy']:.2f}, undecided {e['undecided']:.2f}, thresholds {e['t_benign']:.2f}/{e['t_malicious']:.2f} ({e['status']})")
    L.append("")
    L.append(f"  {'variant':<22}{'poison admitted':>16}{'clean retained':>16}{'backdoor ASR':>14}{'clean bal.acc':>15}")
    for k, m in report["variants"].items():
        L.append(f"  {k:<22}{m['poison_admitted']:>16.2f}{m['clean_retained']:>16.2f}{m['asr']:>14.2f}{m['bal_acc_clean']:>15.2f}")
    L.append("")
    for c in checks:
        L.append(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}  ({c['detail']})")
    ok = all(c["passed"] for c in checks)
    L.append("")
    L.append(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    return "\n".join(L)


def run_demo(seed: int = 0, n_per_class: int = 80, repeats: int = 5, json_path: Optional[str] = None, out=print) -> bool:
    out("ExVCL demo -- SYNTHETIC evidence counts and simulated network features (a library check, not a research result)")
    F, y = synthetic_sessions(n_per_class, seed)
    rep = run_gate_check(F, y, seed, repeats)
    checks = evaluate(rep)
    out(format_report(rep, checks))
    if json_path:
        with open(json_path, "w") as f:
            json.dump(dict(kind="demo", report=rep, checks=checks), f, indent=1)
        out(f"  report written to {json_path}")
    return all(c["passed"] for c in checks)


def _user():
    import getpass
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "?"


def _tool(name):
    p = shutil.which(name)
    return p or "not found"


def run_vm_check(sessions: int = 40, seed: int = 0, workdir: Optional[str] = None, keep: bool = False, json_path: Optional[str] = None,
                 repeats: int = 5, out=print) -> int:
    """Real telemetry on this machine. Returns 0 PASS, 1 FAIL, 2 cannot run (strace missing)."""
    from .session_lab import collect_sessions, strace_available
    out("ExVCL VM check -- REAL strace telemetry from harmless local sessions (a library check, not a research result)")
    env = dict(host=platform.node(), os=" ".join(platform.uname()[:1] + platform.uname()[2:3]), python=sys.version.split()[0],
               user=_user(), strace=_tool("strace"), falco=_tool("falco"), auditctl=_tool("auditctl"), tcpdump=_tool("tcpdump"))
    try:
        env["kernel"] = platform.release()
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            env["ptrace_scope"] = f.read().strip()
    except OSError:
        pass
    for k, val in env.items():
        out(f"  {k}: {val}")
    if not strace_available():
        out("\n  CANNOT RUN: strace is not installed. Install it with:  sudo apt-get install -y strace   (then run this check again)")
        return 2
    wd = workdir or tempfile.mkdtemp(prefix="exvcl_vmcheck_")
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    try:
        F, y, meta = collect_sessions(sessions, seed, wd, progress=lambda i, n: out(f"  recorded {i}/{n} sessions"))
    except Exception as ex:                                          # noqa: BLE001
        out(f"\n  CANNOT RECORD: {type(ex).__name__}: {ex}\n  (a hardened kernel or container can forbid ptrace; check /proc/sys/kernel/yama/ptrace_scope)")
        return 2
    wall = time.time() - t0
    syscalls = F[:, EVENT_KINDS.index("syscall_total")]
    rec_ok = bool((syscalls > 0).all() and all(m["returncode"] in (0, 1, 2, 127) for m in meta))
    out(f"  recording took {wall:.1f} s ({wall / len(y):.2f} s per session); mean syscalls: benign {syscalls[y == 0].mean():.0f}, attack-like {syscalls[y == 1].mean():.0f}")
    means = {k: [round(float(F[y == c][:, i].mean()), 2) for c in (0, 1)] for i, k in enumerate(EVENT_KINDS)}
    out("  mean evidence counts [benign, attack-like]: " + "; ".join(f"{k} {v}" for k, v in means.items()))
    rep = run_gate_check(F, y, seed, repeats)
    checks = evaluate(rep, recording_ok=rec_ok)
    out("")
    out(format_report(rep, checks))
    if json_path:
        with open(json_path, "w") as f:
            json.dump(dict(kind="vm-check", environment=env, evidence_means=means, report=rep, checks=checks, recording_seconds=wall), f, indent=1)
        out(f"  report written to {json_path}")
    if not keep and workdir is None:
        shutil.rmtree(wd, ignore_errors=True)
    else:
        out(f"  strace logs kept in {wd}")
    return 0 if all(c["passed"] for c in checks) else 1

"""exvcl.evalrec -- run the library on a REAL lab recording (run folder from collect_start_p3.sh / collect_stop_p3.sh + sessions.csv).

For every evidence source found (strace, Falco/eBPF, Linux audit) the sessions are split into calibration and stream halves, a share of the
malicious stream sessions is poisoned (label flip + trigger feature on simulated network features), and the gate is run with real evidence
and with the attribution controls. A per-class table shows which kinds of sessions the evidence model reads correctly. This checks the
LIBRARY on real recorded telemetry; it does not replace the pre-specified studies of the paper (those use chronological calibration).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from .checks import evaluate, format_report, run_gate_check
from .evidence import EvidenceModel
from evcl.telemetry import EVENT_KINDS


def _collectors_on_path():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collectors")
    if d not in sys.path:
        sys.path.insert(0, d)


def run_eval_recording(sessions_csv: str, archive: Optional[str] = None, run_dir: Optional[str] = None, sources=("strace", "falco", "auditd"),
                       seed: int = 0, repeats: int = 5, include_hard: bool = False, pad_s: float = 0.5, clock_offset_s: float = 0.0,
                       json_path: Optional[str] = None, out=print) -> int:
    """Returns 0 if every source passes, 1 if any fails, 2 if the recording cannot be read."""
    _collectors_on_path()
    from .collectors.build_dataset_multi import build_multi
    tmp = tempfile.mkdtemp(prefix="exvcl_rec_")
    try:
        if archive:
            with tarfile.open(archive) as t:
                t.extractall(tmp)
            run = tmp
        elif run_dir:
            run = run_dir
        else:
            out("  give --archive RUN.tar.gz or --run-dir FOLDER"); return 2
        try:
            tables, info = build_multi(sessions_csv, run, os.path.join(tmp, "built"), tuple(sources), pad_s, clock_offset_s, with_pcap=False)
        except Exception as e:                                                       # noqa: BLE001
            out(f"  CANNOT READ the recording: {type(e).__name__}: {e}"); return 2
        out(f"ExVCL recording check -- {sessions_csv}")
        for s, i in info.items():
            out(f"  source {s}: {i.get('status')}, events parsed {i.get('events', '-')}, coverage of sessions {i.get('coverage_all', float('nan')):.2f}")
        results, rc = {}, 0
        for src, obs in tables.items():
            obs = obs.copy()
            if not include_hard:
                obs = obs[~obs.cls.astype(str).str.startswith("hard_")]
            obs = obs[obs.covered]
            y = obs.malicious.to_numpy(int)
            F = obs[["ev_" + k for k in EVENT_KINDS]].to_numpy(float)
            out(f"\n=== source: {src}  ({len(y)} covered sessions: {int((y == 0).sum())} benign, {int((y == 1).sum())} malicious)")
            if min((y == 0).sum(), (y == 1).sum()) < 20:
                out("  CANNOT RUN: need at least 20 covered sessions of each label"); rc = max(rc, 2); continue
            rep = run_gate_check(F, y, seed, repeats)
            checks = evaluate(rep)
            out(format_report(rep, checks))
            # per-class table: how the evidence model reads each class (model fitted on a calibration half, read on the other half)
            rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); h = len(y) // 2
            ev = EvidenceModel(seed=seed).fit(F[idx[:h]], y[idx[:h]])
            v = ev.verdicts(F[idx[h:]]).verdict
            tab = pd.DataFrame(dict(cls=obs.cls.to_numpy()[idx[h:]], verdict=v, malicious=y[idx[h:]]))
            per = tab.groupby("cls").apply(lambda g: pd.Series(dict(n=len(g), malicious=int(g.malicious.iloc[0]),
                                                                    read_as_malicious=float((g.verdict == 1).mean()),
                                                                    read_as_benign=float((g.verdict == 0).mean()),
                                                                    undecided=float(g.verdict.isin((2, 3)).mean()))), include_groups=False)
            out("\n  how the evidence reads each class (held-out half):")
            out("  " + per.round(2).to_string().replace("\n", "\n  "))
            results[src] = dict(report=rep, checks=checks, per_class=per.round(3).reset_index().to_dict("records"))
            rc = max(rc, 0 if all(c["passed"] for c in checks) else 1)
        out(f"\nOVERALL: {'PASS' if rc == 0 else ('FAIL' if rc == 1 else 'INCOMPLETE')}")
        if json_path:
            with open(json_path, "w") as f:
                json.dump(dict(kind="eval-recording", sessions=sessions_csv, info=info, results=results), f, indent=1, default=str)
            out(f"report written to {json_path}")
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

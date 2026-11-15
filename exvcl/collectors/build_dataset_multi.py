"""build_dataset_multi -- ONE Phase 3 run folder (strace + Falco + auditd + pcap recorded on the same sessions) -> one observations table per source.

  run folder (made by collect_start_p3.sh / collect_stop_p3.sh): strace.log.gz, falco.jsonl.gz, audit.log.gz, capture.pcap, meta.txt
  sessions.csv (client): session_id, cls, malicious, start_ms, end_ms, params, rc
The pcap features are identical for all sources; only the seven evidence counts (and coverage) differ by source. Labels never enter the evidence."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

try:
    from .audit_parse import parse_audit
    from .falco_parse import parse_falco
    from .strace_parse import EVENT_KINDS, parse_strace, session_counts
except ImportError:  # flat import (scripts, notebooks)
    from audit_parse import parse_audit
    from falco_parse import parse_falco
    from strace_parse import EVENT_KINDS, parse_strace, session_counts

SOURCES = ("strace", "falco", "auditd")


def read_meta(run_dir):
    hits = glob.glob(os.path.join(run_dir, "**", "meta.txt"), recursive=True)      # the archive extracts into an inner folder
    txt = open(hits[0]).read() if hits else ""
    pids = []
    for pat in (r"scope_pids:\s*([\d ]+)", r"traced pids:\s*([\d ]+)", r"sshd_test_pid:\s*(\d+)"):        # collect_start_p3.sh and the Phase 2 collector formats
        m = re.search(pat, txt)
        if m:
            pids += [int(x) for x in m.group(1).split()]
    u = re.search(r"victim_user:\s*(\S+)", txt)
    i = re.search(r"victim_uid:\s*(\d+)", txt)
    return dict(master_pids=sorted(set(pids)), victim_user=u.group(1) if u else "victim",
                victim_uid=int(i.group(1)) if i else None, text=txt)


def _find(run_dir, *names):
    for n in names:
        h = glob.glob(os.path.join(run_dir, "**", n), recursive=True)
        if h:
            return h[0]
    return None


def _audit_span(path):
    import gzip as _gz
    op = _gz.open if path.endswith(".gz") else open
    lo, hi = None, None
    rx = re.compile(r"audit\((\d+\.\d+):")
    with op(path, "rt", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                t = float(m.group(1))
                lo = t if lo is None else min(lo, t)
                hi = t if hi is None else max(hi, t)
    return lo, hi


def _rot_key(f):
    m = re.search(r"\.(\d+)(?:\.gz)?$", os.path.basename(f))
    return -int(m.group(1)) if m else 0            # audit.log.2 (oldest) .. audit.log.1 .. audit.log (newest)


def pick_audit_file(run_dir):
    """The audit evidence may be exported under several names and as a ROTATED SERIES (audit.log.2, audit.log.1, audit.log), and a single file
    can be a partial tail. Candidates: every single audit file, and every rotated series (all audit.log* files of one folder, oldest first).
    Choose the candidate whose records span the longest time; returns (path or list of paths, all candidates)."""
    singles, series = [], {}
    for pat in ("audit*.gz", "audit*.log", "audit*.log.*", "audit*.raw"):
        for f in glob.glob(os.path.join(run_dir, "**", pat), recursive=True):
            b = os.path.basename(f)
            if not os.path.isfile(f) or b.startswith(("auditctl", "auditd_", "audit_rules", "audit_delete", "audit_start", "audit_stop")) or b.endswith((".rules", ".out", ".epoch", ".txt")):
                continue
            if f not in [x for x, *_ in singles]:
                lo, hi = _audit_span(f)
                if lo is not None:
                    singles.append((f, lo, hi))
                    if re.match(r"^audit\.log(\.\d+)?(\.gz)?$", b):        # rotated-series names only
                        series.setdefault(os.path.dirname(f), []).append((f, lo, hi))
    cands = [([f], lo, hi) for f, lo, hi in singles]
    for d, items in series.items():
        if len(items) > 1:
            items = sorted(items, key=lambda x: _rot_key(x[0]))
            cands.append(([x[0] for x in items], min(x[1] for x in items), max(x[2] for x in items)))
    if not cands:
        return None, []
    best = max(cands, key=lambda c: c[2] - c[1])
    chosen = best[0] if len(best[0]) > 1 else best[0][0]
    return chosen, [dict(files=[os.path.basename(p) for p in c[0]], first=c[1], last=c[2]) for c in cands]


def build_multi(sessions_csv, run_dir, out_dir, sources=SOURCES, pad_s=0.5, clock_offset_s=0.0, with_pcap=True):
    os.makedirs(out_dir, exist_ok=True)
    s = pd.read_csv(sessions_csv)
    need = {"session_id", "cls", "malicious", "start_ms", "end_ms"}
    if not need <= set(s.columns) or s.session_id.duplicated().any():
        raise ValueError("bad sessions.csv")
    meta = read_meta(run_dir)
    st, en = s.start_ms.to_numpy(float) / 1000.0, s.end_ms.to_numpy(float) / 1000.0
    base = s[["session_id", "cls", "malicious", "start_ms", "end_ms"]].copy()
    if "rc" in s.columns:
        base["rc"] = s["rc"]
    pcap = _find(run_dir, "capture.pcap")
    if with_pcap and pcap:
        from pcap_features import FEATURES, session_features
        F = session_features(pcap, st, en, None, pad_s, clock_offset_s)
        for j, f in enumerate(FEATURES):
            base["net_" + f] = F[:, j]
    audit_path, audit_cands = pick_audit_file(run_dir)
    paths = dict(strace=_find(run_dir, "strace.log.gz", "strace.log"), falco=_find(run_dir, "falco.jsonl.gz", "falco.jsonl"), auditd=audit_path)
    scope = dict(master_pids=meta["master_pids"], victim_user=meta["victim_user"], victim_uid=meta["victim_uid"])
    if any(x in sources for x in ("falco", "auditd")) and (not scope["master_pids"] or scope["victim_uid"] is None):
        raise ValueError("meta.txt has no scope_pids / victim_uid: the scope of the Falco and audit collectors is unknown. Was the run started with collect_start_p3.sh?")
    out, info = {}, {}
    for src in sources:
        p = paths.get(src)
        if p is None:
            info[src] = dict(status="MISSING FILE")
            continue
        if src == "strace":
            ts, M = parse_strace(p)
        elif src == "falco":
            ts, M = parse_falco(p, scope)
        else:
            ts, M = parse_audit(p, scope)
        cnt = session_counts(ts, M, st, en, pad_s, clock_offset_s)
        obs = base.copy()
        for i, k in enumerate(EVENT_KINDS):
            obs["ev_" + k] = cnt[:, i]
        obs["traced_calls"] = cnt[:, EVENT_KINDS.index("syscall_total")]
        obs["covered"] = obs.traced_calls > 0
        sd = os.path.join(out_dir, src)
        os.makedirs(sd, exist_ok=True)
        obs.to_csv(os.path.join(sd, "observations.csv"), index=False)
        out[src] = obs
        info[src] = dict(status="ok", events=int(len(ts)), **(dict(audit_file=[os.path.basename(x) for x in (p if isinstance(p, list) else [p])], audit_candidates=audit_cands) if src == "auditd" else {}), first=float(ts[0]) if len(ts) else None, last=float(ts[-1]) if len(ts) else None,
                         sessions_outside_span=int(((st < (ts[0] if len(ts) else 0)) | (en > (ts[-1] if len(ts) else 0))).sum()),
                         coverage_all=float(obs.covered.mean()))
    json.dump(dict(sources=info, scope=scope, pad_s=pad_s, clock_offset_s=clock_offset_s, n_sessions=int(len(s))),
              open(os.path.join(out_dir, "build_meta.json"), "w"), indent=2, default=str)
    return out, info
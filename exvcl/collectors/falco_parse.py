"""falco_parse -- Falco JSON events (eBPF) -> per-event evidence matrix with the same seven EVCL event kinds as strace_parse.

Input: the JSON-lines file written by Falco (`file_output`, `json_output: true`) with the experiment rule `evcl_events` (see evcl_falco_rules.yaml.tpl).
Each line carries `time` (ISO, nanoseconds) and `output_fields` with evt.type, proc.pid, proc.ppid, proc.name, proc.exepath, user.name, fd.name,
fd.type, evt.arg.flags. `syscall_total` = number of matched events of the rule's FIXED event list (not all syscalls).
Scope: events are kept if they belong to the experiment services: the process runs as the lab user, or its parent chain (built from the
events themselves) reaches one of the experiment master pids. Pass scope=None to keep everything."""
from __future__ import annotations

import calendar
import gzip
import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from .strace_parse import EVENT_KINDS, _PRIV, _SENSITIVE
except ImportError:  # flat import (scripts, notebooks)
    from strace_parse import EVENT_KINDS, _PRIV, _SENSITIVE

_K = {k: i for i, k in enumerate(EVENT_KINDS)}
_ISO = re.compile(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?Z?$")
_WRITE = re.compile(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND")
_SHELLS = {"bash", "sh", "dash", "zsh", "ash", "rbash", "ksh", "csh"}
_OPEN = {"open", "openat", "openat2", "creat"}


def iso_to_epoch(s: str) -> Optional[float]:
    m = _ISO.match(s.strip())
    if not m:
        return None
    y, mo, d, h, mi, se, frac = m.groups()
    return calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(se))) + (float("0." + frac) if frac else 0.0)


def _lines(src) -> Iterable[str]:
    if isinstance(src, str) and "\n" not in src and len(src) < 4096 and os.path.exists(src):
        op = gzip.open if src.endswith(".gz") else open
        with op(src, "rt", errors="replace") as f:
            yield from f
    elif isinstance(src, str):
        yield from src.splitlines()
    else:
        yield from src


def classify_event(et: str, f: Dict) -> Optional[list]:
    """kind indices triggered by one Falco event (always includes syscall_total), or None to ignore."""
    if not et:
        return None
    out = [_K["syscall_total"]]
    if et in ("execve", "execveat"):
        out.append(_K["process_spawn"])
        exe = str(f.get("proc.exepath") or f.get("evt.arg.exe") or f.get("proc.name") or "")
        if os.path.basename(exe) in _SHELLS:
            out.append(_K["shell_exec"])
    elif et in _OPEN:
        if et == "creat" or _WRITE.search(str(f.get("evt.arg.flags") or "")):
            out.append(_K["file_write"])
        path = str(f.get("fd.name") or f.get("evt.arg.name") or f.get("evt.arg.path") or "")
        if any(s in path for s in _SENSITIVE):
            out.append(_K["sensitive_file_access"])
    elif et == "connect":
        ft = str(f.get("fd.type") or "")
        if ft in ("ipv4", "ipv6") or ("->" in str(f.get("fd.name") or "") and "/" not in str(f.get("fd.name") or "")):
            out.append(_K["net_connect_out"])
    elif et in _PRIV:
        out.append(_K["priv_change"])
    elif et in ("accept", "accept4", "bind", "listen", "clone", "clone3", "fork", "vfork"):
        pass
    else:
        return None
    return out


def scope_mask(pids, ppids, users, master_pids, victim_user) -> np.ndarray:
    """True for events of processes that run as the lab user or descend (through the observed pid->ppid links) from an experiment master pid."""
    pp: Dict[int, int] = {}
    for p, q in zip(pids, ppids):
        if p >= 0 and q >= 0:
            pp[p] = q
    memo: Dict[int, bool] = {}
    masters = set(int(x) for x in master_pids)

    def lineage(p: int) -> bool:
        if p in memo:
            return memo[p]
        chain, cur, ok = [], p, False
        for _ in range(16):
            if cur in masters:
                ok = True
                break
            chain.append(cur)
            if cur not in pp or pp[cur] in (0, cur):
                break
            cur = pp[cur]
        for c in chain:
            memo[c] = ok
        return ok
    return np.array([(victim_user is not None and u == victim_user) or lineage(p) for p, u in zip(pids, users)], bool)


def parse_falco(src, scope: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """-> (ts seconds float64 [n], M int16 [n, 7]) sorted by time. scope = {"master_pids": [...], "victim_user": "victim"} or None."""
    ts, rows, pids, ppids, users = [], [], [], [], []
    for raw in _lines(src):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        f = r.get("output_fields") or {}
        t = iso_to_epoch(str(r.get("time", "")))
        if t is None and f.get("evt.rawtime") is not None:
            t = float(f["evt.rawtime"]) / 1e9
        kinds = classify_event(str(f.get("evt.type") or ""), f)
        if t is None or kinds is None:
            continue
        v = np.zeros(len(EVENT_KINDS), dtype=np.int16)
        v[kinds] = 1
        ts.append(t); rows.append(v)
        pids.append(int(f.get("proc.pid", -1))); ppids.append(int(f.get("proc.ppid", -1))); users.append(f.get("user.name"))
    if not ts:
        return np.zeros(0), np.zeros((0, len(EVENT_KINDS)), dtype=np.int16)
    ts, M = np.array(ts), np.vstack(rows)
    if scope:
        keep = scope_mask(pids, ppids, users, scope.get("master_pids", []), scope.get("victim_user"))
        ts, M = ts[keep], M[keep]
    o = np.argsort(ts, kind="stable")
    return ts[o], M[o]
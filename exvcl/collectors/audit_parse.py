"""audit_parse -- Linux audit log (auditd, raw audit.log) -> per-event evidence matrix with the same seven EVCL event kinds as strace_parse.

Records of one event share the id in `msg=audit(<epoch>:<serial>)`: SYSCALL (syscall number, args, pid/ppid/uid, exe), PATH (file names),
SOCKADDR (connect address). x86_64 only. `syscall_total` = number of audited events of the FIXED rule list (see evcl_audit.rules.tpl).
Scope: same rule as falco_parse (lab-user uid, or descendants of an experiment master pid through the observed pid/ppid links)."""
from __future__ import annotations

import gzip
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from .falco_parse import _SHELLS, scope_mask
    from .strace_parse import EVENT_KINDS, _SENSITIVE
except ImportError:  # flat import (scripts, notebooks)
    from falco_parse import _SHELLS, scope_mask
    from strace_parse import EVENT_KINDS, _SENSITIVE

_K = {k: i for i, k in enumerate(EVENT_KINDS)}
_MSG = re.compile(r"msg=audit\((\d+\.\d+):(\d+)\)")
_KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
X64 = {2: "open", 257: "openat", 437: "openat2", 85: "creat", 59: "execve", 322: "execveat", 42: "connect", 43: "accept", 288: "accept4",
       49: "bind", 50: "listen", 56: "clone", 57: "fork", 58: "vfork", 435: "clone3", 105: "setuid", 106: "setgid", 113: "setreuid",
       114: "setregid", 117: "setresuid", 119: "setresgid", 126: "capset", 161: "chroot", 165: "mount"}
_PRIV = {"setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "capset", "chroot", "mount"}
_WRITE_MASK = 0x1 | 0x2 | 0x40 | 0x200 | 0x400          # O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND


def _lines(src) -> Iterable[str]:
    if isinstance(src, (list, tuple)) and src and all(isinstance(p, str) and "\n" not in p and os.path.exists(p) for p in src):
        for p in src:                                   # a rotated series: oldest first
            yield from _lines(p)
    elif isinstance(src, str) and "\n" not in src and len(src) < 4096 and os.path.exists(src):
        op = gzip.open if src.endswith(".gz") else open
        with op(src, "rt", errors="replace") as f:
            yield from f
    elif isinstance(src, str):
        yield from src.splitlines()
    else:
        yield from src


def _kv(line: str) -> Dict[str, str]:
    return {k: v.strip('"') for k, v in _KV.findall(line)}


def _hex(x: str) -> int:
    try:
        return int(x, 16)
    except (TypeError, ValueError):
        return 0


def classify_audit(sc: Dict, paths, saddr: Optional[str]) -> Optional[list]:
    name = X64.get(int(sc.get("syscall", -1)))
    if name is None or sc.get("arch", "c000003e").lower() != "c000003e":
        return None
    out = [_K["syscall_total"]]
    if name in ("execve", "execveat"):
        out.append(_K["process_spawn"])
        if os.path.basename(sc.get("exe", "")) in _SHELLS:
            out.append(_K["shell_exec"])
    elif name in ("open", "openat", "openat2", "creat"):
        flags = _hex(sc.get("a1" if name == "open" else "a2", "0")) if name not in ("creat", "openat2") else 0
        if name == "creat" or (flags & _WRITE_MASK):
            out.append(_K["file_write"])
        if any(s in p for p in paths for s in _SENSITIVE):
            out.append(_K["sensitive_file_access"])
    elif name == "connect":
        fam = (saddr or "")[:4].upper()
        if fam in ("0200", "0A00"):                            # AF_INET / AF_INET6 (little-endian family field)
            out.append(_K["net_connect_out"])
    elif name in _PRIV:
        out.append(_K["priv_change"])
    return out


def parse_audit(src, scope: Optional[dict] = None, victim_uid: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """-> (ts [n], M [n, 7]) sorted by time. scope = {"master_pids": [...], "victim_uid": 1001} (or pass victim_uid)."""
    events, cur_id, cur = [], None, {"sc": None, "paths": [], "saddr": None}

    def flush():
        if cur["sc"] is not None:
            events.append((cur_id, cur["sc"], list(cur["paths"]), cur["saddr"]))
    for raw in _lines(src):
        m = _MSG.search(raw)
        if not m:
            continue
        eid = (m.group(1), m.group(2))
        if eid != cur_id:
            flush()
            cur_id, cur = eid, {"sc": None, "paths": [], "saddr": None}
        kv = _kv(raw)
        if raw.startswith("type=SYSCALL"):
            cur["sc"] = kv
        elif raw.startswith("type=PATH") and "name" in kv:
            cur["paths"].append(kv["name"])
        elif raw.startswith("type=SOCKADDR"):
            cur["saddr"] = kv.get("saddr")
    flush()
    ts, rows, pids, ppids, uids = [], [], [], [], []
    for (t, _), sc, paths, saddr in events:
        kinds = classify_audit(sc, paths, saddr)
        if kinds is None:
            continue
        v = np.zeros(len(EVENT_KINDS), dtype=np.int16)
        v[kinds] = 1
        ts.append(float(t)); rows.append(v)
        pids.append(int(sc.get("pid", -1))); ppids.append(int(sc.get("ppid", -1))); uids.append(int(sc.get("uid", -2)))
    if not ts:
        return np.zeros(0), np.zeros((0, len(EVENT_KINDS)), dtype=np.int16)
    ts, M = np.array(ts), np.vstack(rows)
    vu = (scope or {}).get("victim_uid", victim_uid)
    if scope or victim_uid is not None:
        keep = scope_mask(pids, ppids, [u if (vu is not None and u == int(vu)) else None for u in uids],
                          (scope or {}).get("master_pids", []), None) | (np.array(uids) == int(vu) if vu is not None else False)
        ts, M = ts[keep], M[keep]
    o = np.argsort(ts, kind="stable")
    return ts[o], M[o]
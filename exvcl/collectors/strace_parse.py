"""strace_parse -- real strace logs -> per-session evidence counts (the seven EVCL event kinds).

Input: `strace -f -ttt` output (one timestamped log for a whole run). Output: per-line event matrix, then counts per session window.
The event kinds and their meaning are identical to evcl.telemetry (process_spawn, shell_exec, file_write, sensitive_file_access,
net_connect_out, priv_change, syscall_total). `syscall_total` here = number of TRACED calls (the collector traces a fixed list).
Handles: pid prefixes ('1234  ' or '[pid 1234] '), '-ttt' epoch timestamps, '<unfinished ...>' / '<... resumed>' pairs (a call is
counted once, at its unfinished line), '+++ exited' and '--- SIG' lines (ignored), optional gzip input."""
from __future__ import annotations

import gzip
import re
from typing import Iterable, Tuple

import numpy as np

EVENT_KINDS = ("process_spawn", "shell_exec", "file_write", "sensitive_file_access", "net_connect_out", "priv_change", "syscall_total")
_K = {k: i for i, k in enumerate(EVENT_KINDS)}
_LINE = re.compile(r"^(?:\[pid\s+\d+\]\s+|\d+\s+)?(\d+\.\d+)\s+(.*)$")
_CALL = re.compile(r"(\w+)\(")
_SENSITIVE = ("/etc/shadow", "/etc/passwd", "/etc/sudoers", ".ssh/", "/proc/self/mem", "/root/")
_PRIV = {"setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "capset", "chroot", "mount"}
_SHELL = re.compile(r'"/(?:usr/)?(?:s?bin)/(?:ba|da|z|a)?sh"')
_WRITE_FLAGS = re.compile(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND")
_OPENERS = {"open", "openat", "openat2", "creat"}


def _lines(src) -> Iterable[str]:
    if isinstance(src, str) and "\n" not in src and len(src) < 4096:
        op = gzip.open if src.endswith(".gz") else open
        with op(src, "rt", errors="replace") as f:
            yield from f
    elif isinstance(src, str):
        yield from src.splitlines()
    else:
        yield from src


def classify_call(rest: str):
    """Return the list of event-kind indices triggered by one syscall line (always includes syscall_total) or None to ignore the line."""
    if rest.startswith("<...") or rest.startswith("+++") or rest.startswith("---"):
        return None
    m = _CALL.match(rest)
    if not m:
        return None
    name = m.group(1)
    out = [_K["syscall_total"]]
    if name in ("execve", "execveat"):
        out.append(_K["process_spawn"])
        if _SHELL.search(rest):
            out.append(_K["shell_exec"])
    elif name in _OPENERS:
        if name == "creat" or _WRITE_FLAGS.search(rest):
            out.append(_K["file_write"])
        if any(s in rest for s in _SENSITIVE):
            out.append(_K["sensitive_file_access"])
    elif name == "connect" and ("AF_INET" in rest or "AF_INET6" in rest):
        out.append(_K["net_connect_out"])
    elif name in _PRIV:
        out.append(_K["priv_change"])
    return out


def parse_strace(src) -> Tuple[np.ndarray, np.ndarray]:
    """-> (ts seconds float64 [n], M int16 [n, 7]) sorted by time; src = path (.gz ok), text, or iterable of lines."""
    ts, rows = [], []
    for raw in _lines(src):
        m = _LINE.match(raw.strip())
        if not m:
            continue
        kinds = classify_call(m.group(2))
        if kinds is None:
            continue
        v = np.zeros(len(EVENT_KINDS), dtype=np.int16)
        v[kinds] = 1
        ts.append(float(m.group(1)))
        rows.append(v)
    if not ts:
        return np.zeros(0), np.zeros((0, len(EVENT_KINDS)), dtype=np.int16)
    ts = np.array(ts)
    M = np.vstack(rows)
    o = np.argsort(ts, kind="stable")
    return ts[o], M[o]


def session_counts(ts: np.ndarray, M: np.ndarray, starts_s, ends_s, pad_s: float = 0.5, clock_offset_s: float = 0.0) -> np.ndarray:
    """Counts of each event kind inside every [start - pad, end + pad] window (+ offset between the client's and the target's clock)."""
    starts = np.asarray(starts_s, float) + clock_offset_s - pad_s
    ends = np.asarray(ends_s, float) + clock_offset_s + pad_s
    cum = np.vstack([np.zeros((1, M.shape[1]), dtype=np.int64), np.cumsum(M, axis=0, dtype=np.int64)]) if len(ts) else np.zeros((1, len(EVENT_KINDS)), dtype=np.int64)
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, ends, side="right")
    return cum[hi] - cum[lo]


def estimate_clock_offset(ts, M, starts_s, ends_s, active_mask=None, search_s: float = 2.0, step_s: float = 0.1, pad_s: float = 0.0):
    """Diagnostic: if the client's and the target's clocks differ, evidence lands in the wrong window. Scan offsets in [-search, +search] and
    return the one that puts the MOST traced calls inside the active sessions' windows (idle windows are ignored). NOT applied automatically."""
    starts, ends = np.asarray(starts_s, float), np.asarray(ends_s, float)
    if active_mask is not None:
        starts, ends = starts[active_mask], ends[active_mask]
    offs = np.arange(-search_s, search_s + 1e-9, step_s)
    vals = np.array([session_counts(ts, M, starts, ends, pad_s=pad_s, clock_offset_s=o)[:, EVENT_KINDS.index("syscall_total")].sum() for o in offs], float)
    if vals.max() <= 0:
        return 0.0
    plateau = offs[vals >= 0.98 * vals.max()]                      # the best offsets form a plateau when the window is wider than the burst
    return float(np.median(plateau))
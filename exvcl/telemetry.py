"""exvcl.telemetry -- the seven evidence counts and helpers to obtain them from real logs."""
from __future__ import annotations

import numpy as np

from evcl.telemetry import EVENT_KINDS, TelemetryEvent, normalize_events, parse_strace

__all__ = ["EVENT_KINDS", "TelemetryEvent", "counts_from_events", "counts_from_strace_text", "counts_from_strace_file"]


def counts_from_events(events) -> np.ndarray:
    """Normalized events -> count vector in the order of EVENT_KINDS."""
    return normalize_events(events)


def counts_from_strace_text(text: str) -> np.ndarray:
    """Text of `strace -f` (one execution) -> the seven evidence counts (process_spawn, shell_exec, file_write,
    sensitive_file_access, net_connect_out, priv_change, syscall_total)."""
    return normalize_events(parse_strace(text))


def counts_from_strace_file(path: str) -> np.ndarray:
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        return counts_from_strace_text(f.read())

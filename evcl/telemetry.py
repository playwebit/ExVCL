"""evcl.telemetry -- normalized behavioural events and parsers."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

EVENT_KINDS = ("process_spawn", "shell_exec", "file_write", "sensitive_file_access",
               "net_connect_out", "priv_change", "syscall_total")


@dataclass(frozen=True)
class TelemetryEvent:
    kind: str            # one of EVENT_KINDS
    name: str = ""
    count: int = 1


def normalize_events(events: Iterable[TelemetryEvent]) -> np.ndarray:
    """Events -> fixed-order raw count vector (len == len(EVENT_KINDS))."""
    c = Counter()
    for e in events:
        if e.kind in EVENT_KINDS:
            c[e.kind] += int(e.count)
    return np.array([c[k] for k in EVENT_KINDS], dtype=float)


_SENSITIVE = ("/etc/shadow", "/etc/passwd", "/etc/sudoers", ".ssh/", "/proc/self/mem", "/root/")
_PRIV = {"setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "capset", "chroot", "mount"}
_SHELL = re.compile(r'"/(?:usr/)?(?:s?bin)/(?:ba|da|z)?sh"')


def parse_strace(text: str) -> List[TelemetryEvent]:
    """Parse `strace -f` text into normalized events (best effort, heuristic)."""
    c = Counter()
    for raw in text.splitlines():
        line = re.sub(r"^\[pid\s+\d+\]\s*|^\d+\s+", "", raw.strip())
        m = re.match(r"(\w+)\(", line)
        if not m:
            continue
        name = m.group(1)
        c["syscall_total"] += 1
        if name in ("execve", "execveat"):
            c["process_spawn"] += 1
            if _SHELL.search(line):
                c["shell_exec"] += 1
        elif name in ("open", "openat", "openat2", "creat"):
            if name == "creat" or re.search(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC", line):
                c["file_write"] += 1
            if any(s in line for s in _SENSITIVE):
                c["sensitive_file_access"] += 1
        elif name == "connect" and ("AF_INET" in line or "AF_INET6" in line):
            c["net_connect_out"] += 1
        elif name in _PRIV:
            c["priv_change"] += 1
    return [TelemetryEvent(k, "strace", v) for k, v in c.items() if v > 0]
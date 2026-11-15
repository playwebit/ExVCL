"""exvcl.session_lab -- harmless local sessions recorded with REAL strace, for the VM check.

Every session is a short command run as `strace -f -o LOG -- COMMAND` by the current user, inside a throw-away work folder:
benign sessions write and read temp files, print system facts or make one local connection; attack-like sessions probe credential
files (the reads are denied or the files do not exist), chain shells, or burst failed connections to CLOSED ports on 127.0.0.1.
Nothing leaves the machine, nothing is modified outside the work folder, no privileges are needed. Labels follow the generator's intent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Dict, List, Tuple

import numpy as np

from .telemetry import counts_from_strace_text

BENIGN_KINDS = ("files", "pyjson", "netone", "sysinfo", "idle", "admin_read")     # label 0 ("admin_read" overlaps with probing on purpose)
ATTACK_KINDS = ("credprobe", "shellchain", "connburst", "mixprobe")                # label 1
_BENIGN_W = (0.22, 0.2, 0.2, 0.18, 0.1, 0.1)

_PY_CONNECT = ("import socket,sys\n"
               "n=int(sys.argv[1])\n"
               "for p in range(n):\n"
               "    s=socket.socket(); s.settimeout(0.05)\n"
               "    try: s.connect(('127.0.0.1', 9+p))\n"
               "    except OSError: pass\n"
               "    s.close()\n")
_SH_PROBE = ('r=0; while [ "$r" -lt "$1" ]; do for f in /etc/shadow /etc/sudoers /root/.ssh/id_rsa "$HOME/.ssh/id_rsa" /proc/self/mem; '
             'do cat "$f" >/dev/null 2>&1; done; r=$((r+1)); done')
_SH_CHAIN = 'i=0; while [ "$i" -lt "$1" ]; do sh -c "id -u" >/dev/null; sh -c "uname -s" >/dev/null; i=$((i+1)); done'
_SH_FILES = ('d=$1; mkdir -p "$d"; i=0; while [ "$i" -lt "$2" ]; do echo line$i >> "$d/f"; i=$((i+1)); done; '
             'cat "$d/f" | wc -l >/dev/null; ls "$d" >/dev/null')


def strace_available() -> bool:
    return shutil.which("strace") is not None


def _command(kind: str, rng: np.random.Generator, wd: str, n: int) -> List[str]:
    sub = os.path.join(wd, f"w{n}")
    if kind == "files":
        return ["sh", "-c", _SH_FILES, "sh", sub, str(int(rng.integers(3, 13)))]
    if kind == "pyjson":
        return ["python3", "-c", "import json,sys;json.dump(list(range(int(sys.argv[2]))),open(sys.argv[1],'w'))",
                os.path.join(wd, f"j{n}.json"), str(int(rng.integers(10, 200)))]
    if kind == "netone":
        return ["python3", "-c", _PY_CONNECT, "1"]
    if kind == "sysinfo":
        return ["sh", "-c", "uname -s >/dev/null; date >/dev/null; id -u >/dev/null; df / >/dev/null 2>&1"]
    if kind == "idle":
        return ["sleep", "0.1"]
    if kind == "admin_read":                       # legitimate admin look at the user database (overlaps with credential probing)
        return ["sh", "-c", "cat /etc/passwd >/dev/null; ls /home >/dev/null; df / >/dev/null 2>&1"]
    if kind == "credprobe":
        return ["sh", "-c", _SH_PROBE, "sh", str(int(rng.integers(1, 4)))]
    if kind == "shellchain":
        return ["sh", "-c", _SH_CHAIN, "sh", str(int(rng.integers(3, 9)))]
    if kind == "connburst":
        return ["python3", "-c", _PY_CONNECT, str(int(rng.integers(8, 25)))]
    if kind == "mixprobe":
        return ["sh", "-c", f'python3 -c \'{_PY_CONNECT}\' {int(rng.integers(4, 9))}; {_SH_PROBE.replace("$1", "1")}; sh -c "id -u" >/dev/null; sh -c "uname -s" >/dev/null']
    raise ValueError(kind)


def record_session(kind: str, rng: np.random.Generator, workdir: str, n: int = 0, timeout: float = 60.0) -> Tuple[np.ndarray, Dict]:
    """Run one session under strace and return (7 evidence counts, metadata)."""
    if not strace_available():
        raise RuntimeError("strace is not installed (sudo apt-get install -y strace)")
    log = os.path.join(workdir, f"{n:04d}_{kind}.strace")
    cmd = _command(kind, rng, workdir, n)
    t0 = time.time()
    r = subprocess.run(["strace", "-f", "-o", log, "--"] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, cwd=workdir)
    wall = time.time() - t0
    with open(log, errors="replace") as f:
        text = f.read()
    return counts_from_strace_text(text), dict(kind=kind, wall_s=round(wall, 3), returncode=r.returncode, log_bytes=len(text), log=log)


def collect_sessions(n_per_class: int, seed: int, workdir: str, progress=None):
    """Record n_per_class benign and n_per_class attack-like sessions (shuffled order). Returns F (n,7), y (n,), meta list."""
    if n_per_class < 20:
        raise ValueError("need at least 20 sessions per class (10 per class are used for calibration)")
    rng = np.random.default_rng(seed)
    plan = [(k, 0) for k in rng.choice(BENIGN_KINDS, n_per_class, p=_BENIGN_W)] + [(k, 1) for k in rng.choice(ATTACK_KINDS, n_per_class)]
    order = rng.permutation(len(plan))
    F, y, meta = [], [], []
    os.makedirs(workdir, exist_ok=True)
    for i, j in enumerate(order):
        kind, lab = plan[j]
        f, m = record_session(str(kind), rng, workdir, i)
        F.append(f); y.append(lab); meta.append(m)
        if progress and (i + 1) % 10 == 0:
            progress(i + 1, len(plan))
    return np.vstack(F), np.array(y, int), meta

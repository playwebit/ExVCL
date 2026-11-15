"""Sandboxed execution of a request under strace (with -s 2048: strace truncates strings at 32 characters by default, which would cut
long traversal paths and silently hide the sensitive part) -> the seven evidence counts. The worker runs the vulnerable handler in a separate
process (`python3 -I -S target.py NAME`); only that process is traced, so the evidence is what this request's execution really does."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..telemetry import EVENT_KINDS, counts_from_strace_text
from . import target
from .payloads import server_decode

TARGET = os.path.abspath(target.__file__)


def run_sandbox(wire_name: str, workdir: str, tag: str = "x", timeout: float = 30.0):
    """Returns (F (7,), ok, seconds). ok is False if the execution produced no telemetry."""
    log = os.path.join(workdir, f"{tag}.strace")
    t0 = time.time()
    try:
        subprocess.run(["strace", "-f", "-s", "2048", "-o", log, "--", sys.executable, "-I", "-S", TARGET, server_decode(wire_name)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        with open(log, errors="replace") as f:
            F = counts_from_strace_text(f.read())
        ok = bool(F[EVENT_KINDS.index("syscall_total")] > 0)
    except Exception:                                                           # noqa: BLE001
        F, ok = np.zeros(len(EVENT_KINDS)), False
    finally:
        try:
            os.remove(log)
        except OSError:
            pass
    return F, ok, time.time() - t0


def collect_evidence(wire_names, workers: int = 4, progress=None, workdir: Optional[str] = None):
    """Run every request in the sandbox (parallel). Returns F (n,7), ok (n,), seconds (n,)."""
    if shutil.which("strace") is None:
        raise RuntimeError("strace is not installed (sudo apt-get install -y strace)")
    target.ensure_lab()
    wd = workdir or tempfile.mkdtemp(prefix="exvcl_weblab_")
    os.makedirs(wd, exist_ok=True)
    done = [0]

    def one(i):
        r = run_sandbox(wire_names[i], wd, f"s{i}")
        done[0] += 1
        if progress and done[0] % 50 == 0:
            progress(done[0], len(wire_names))
        return r
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one, range(len(wire_names))))
    if workdir is None:
        shutil.rmtree(wd, ignore_errors=True)
    return np.vstack([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res])

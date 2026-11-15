"""evcl.sandbox -- sandbox adapter interface and implementations.

IMPORTANT HONESTY NOTES
  * SimulatedSandbox produces MODELED telemetry from assumed distributions. It is for validating the
    pipeline logic and for sensitivity analysis. It is not evidence about real sandboxes.
  * ContainerSandboxAdapter / RecordedTelemetrySandbox produce telemetry from real executions / real
    recordings, but ContainerSandboxAdapter has NOT been validated here and Docker is not a strong
    isolation boundary for real malware.
  * VMSandboxAdapter is an interface stub.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .core import EnvironmentNotSupported, docker_probe
from .data import OracleChannel
from .telemetry import EVENT_KINDS, TelemetryEvent, parse_strace


@dataclass
class SandboxRequest:
    obs_id: int
    payload_ref: str                       # how a real adapter locates the artefact to execute (NO ground truth here)


@dataclass
class SandboxResult:
    obs_id: int
    status: str                            # "ok" | "timeout" | "error"
    events: List[TelemetryEvent]
    wall_seconds: float
    mode: str
    is_simulated: bool
    error: str = ""


class SandboxAdapter(ABC):
    mode: str = "abstract"
    source: str = "ABSTRACT"               # telemetry provenance tag carried into every result table
    is_simulated: bool = False
    n_executed: int = 0

    @abstractmethod
    def execute(self, req: SandboxRequest) -> SandboxResult: ...

    def execute_many(self, reqs: List[SandboxRequest]) -> List[SandboxResult]:
        self.n_executed += len(reqs)
        return [self.execute(r) for r in reqs]

    def describe(self) -> Dict[str, Any]:
        return dict(mode=self.mode, is_simulated=self.is_simulated)


# ----------------------------------------------------------------------------- simulated
_BENIGN = dict(process_spawn=1.0, shell_exec=0.1, file_write=3.0, sensitive_file_access=0.05,
               net_connect_out=2.0, priv_change=0.02, syscall_total=60.0)
_MALIC = dict(process_spawn=4.0, shell_exec=1.5, file_write=8.0, sensitive_file_access=1.5,
              net_connect_out=6.0, priv_change=0.8, syscall_total=180.0)


class SimulatedSandbox(SandboxAdapter):
    """MODELED telemetry: Poisson counts whose rates interpolate between hand-set benign and malicious
    profiles. Evasive malicious samples behave benignly with probability `detectability`."""
    mode = "SIMULATED"
    source = "SIMULATED"
    is_simulated = True

    def __init__(self, rng: np.random.Generator, signal=1.0, detectability=0.5, benign_noise=0.03):
        self.rng, self.signal, self.detectability, self.benign_noise = rng, signal, detectability, benign_noise
        self._oracle: Optional[OracleChannel] = None

    def set_oracle(self, oracle: OracleChannel):
        """The ONLY ground-truth path into the pipeline: simulated telemetry is generated FROM ground truth.
        Consequently simulated verification is NOT independent of the evaluation labels."""
        self._oracle = oracle

    def execute(self, req: SandboxRequest) -> SandboxResult:
        if self._oracle is None:
            raise ValueError("SimulatedSandbox needs an OracleChannel (simulation-only ground truth).")
        t0 = time.perf_counter()
        mal, evasive = self._oracle.lookup(req.obs_id)
        eff = self.signal if mal else 0.0
        if mal and evasive and self.rng.random() < self.detectability:
            eff = 0.0                                  # sandbox detected -> behaviour suppressed
        if (not mal) and self.rng.random() < self.benign_noise:
            eff = 0.6 * self.signal                    # noisy benign (e.g. admin tooling)
        events = []
        for k in EVENT_KINDS:
            rate = _BENIGN[k] + eff * (_MALIC[k] - _BENIGN[k])
            n = int(self.rng.poisson(rate))
            if n > 0:
                events.append(TelemetryEvent(k, f"sim_{k}", n))
        return SandboxResult(req.obs_id, "ok", events, time.perf_counter() - t0, self.mode, True)

    def describe(self):
        return dict(super().describe(), signal=self.signal, detectability=self.detectability,
                    benign_noise=self.benign_noise, warning="telemetry is MODELED, not measured")


class SimulatedBehaviorSandbox(SandboxAdapter):
    """SIMULATED v2: telemetry is a stochastic function of the observation's TRUE behavioural family, degraded by
    explicit fidelity knobs. This is NOT independent of ground truth (simulation cannot be); it bounds how informative
    evidence is. Knobs: signal (0 => independent of truth), overlap (dormant malware), benign_suspicious (admin-like
    benign), telemetry_loss (per-event drop), window_frac (fraction of malicious activity observed), delay_prob
    (activation after the window), failure_rate (execution fails -> no evidence), and per-observation evasion latents
    (env-sensitive / delayed / mimicry / partial). Event rates are hand-set, NOT measured from real malware."""
    mode = "SIMULATED"
    source = "SIMULATED"
    is_simulated = True

    def __init__(self, rng: np.random.Generator, sc):
        self.rng, self.sc = rng, sc
        self._oracle: Optional[OracleChannel] = None

    def set_oracle(self, oracle: OracleChannel):
        self._oracle = oracle

    def _effect(self, cls: int, evasive: bool, mode: int) -> float:
        sc, r = self.sc, self.rng
        if sc.signal <= 0:
            return 0.0                                        # null: telemetry independent of ground truth
        if cls == 0:
            return 0.6 * sc.signal if r.random() < sc.benign_suspicious else 0.0
        eff = sc.signal * (0.7 + 0.3 * (((cls * 37) % 10) / 9.0))   # family-specific strength
        if r.random() < sc.overlap or r.random() < sc.delay_prob:
            return 0.0                                        # dormant / activation after the window
        if evasive:
            if mode in (0, 1):
                if r.random() < sc.detectability:             # environment-sensitive: sandbox detected
                    return 0.0
            elif mode == 2:
                if r.random() < sc.evasion_delay_prob:        # deliberately delayed activation
                    return 0.0
            elif mode == 3:
                eff *= (1.0 - sc.mimicry)                     # decoy / mimicry of benign activity
            elif mode == 4:
                eff *= 0.3                                    # low-and-slow partial activity
        return eff * sc.window_frac

    def execute(self, req: SandboxRequest) -> SandboxResult:
        if self._oracle is None:
            raise ValueError("SimulatedBehaviorSandbox needs an OracleChannel (simulation-only ground truth).")
        t0 = time.perf_counter()
        cls, evasive, mode = self._oracle.lookup_full(req.obs_id)
        if self.rng.random() < self.sc.failure_rate:
            return SandboxResult(req.obs_id, "timeout", [], time.perf_counter() - t0, self.mode, True, "simulated failure")
        eff = self._effect(cls, evasive, mode)
        events = []
        for k in EVENT_KINDS:
            n = int(self.rng.poisson(_BENIGN[k] + eff * (_MALIC[k] - _BENIGN[k])))
            if self.sc.telemetry_loss > 0 and n > 0:
                n = int(self.rng.binomial(n, 1.0 - self.sc.telemetry_loss))
            if n > 0:
                events.append(TelemetryEvent(k, f"sim_{k}", n))
        return SandboxResult(req.obs_id, "ok", events, time.perf_counter() - t0, self.mode, True)

    def describe(self):
        return dict(super().describe(), model="v2_behavior", **{k: getattr(self.sc, k) for k in (
            "signal", "overlap", "benign_suspicious", "telemetry_loss", "window_frac", "delay_prob", "failure_rate",
            "detectability", "evasion_delay_prob", "mimicry")}, warning="telemetry is MODELED from true behaviour")


class _ControlWrapper(SandboxAdapter):
    """Negative-control wrapper. Calibration runs through the REAL inner adapter; the control only becomes active
    at deployment (activate_control), i.e. a normally-calibrated gate then receives degraded evidence."""

    def __init__(self, inner: SandboxAdapter):
        self.inner = inner
        self.mode = inner.mode
        self.source = inner.source
        self.is_simulated = inner.is_simulated
        self.active = False

    def set_oracle(self, oracle):
        if hasattr(self.inner, "set_oracle"):
            self.inner.set_oracle(oracle)

    def activate_control(self):
        self.active = True

    def execute(self, req):
        return self.execute_many([req])[0]


class ShuffledEvidenceSandbox(_ControlWrapper):
    """CONTROL: after activation, telemetry is randomly permuted across the observations of each batch
    (marginal distribution preserved, association with the observation destroyed)."""

    def __init__(self, inner, rng):
        super().__init__(inner)
        self.rng = rng

    def execute_many(self, reqs):
        res = self.inner.execute_many(reqs)
        self.n_executed += len(reqs)
        if self.active:
            oks = [i for i, r in enumerate(res) if r.status == "ok"]
            ev = [res[i].events for i in oks]
            perm = self.rng.permutation(len(oks))
            for pos, i in enumerate(oks):
                res[i] = SandboxResult(res[i].obs_id, "ok", ev[perm[pos]], res[i].wall_seconds, res[i].mode,
                                       res[i].is_simulated)
        return res


class GlobalRandomEvidenceSandbox(_ControlWrapper):
    """CONTROL (stronger null than ShuffledEvidenceSandbox): after activation every observation receives telemetry drawn at random from a
    pool collected during the CALIBRATION period (pre-activation executions of the real adapter). The evidence is independent of the
    observation, of its class AND of triage routing, and carries the stream-wide (calibration-period) base rates instead of the base
    rate of the already-triaged batch. If the gate still protects with this evidence, the protection does not come from evidence content."""

    def __init__(self, inner, rng, pool_cap=4000):
        super().__init__(inner)
        self.rng = rng
        self.pool = []
        self.pool_cap = int(pool_cap)

    def execute_many(self, reqs):
        res = self.inner.execute_many(reqs)
        self.n_executed += len(reqs)
        oks = [i for i, r in enumerate(res) if r.status == "ok"]
        if not self.active:
            for i in oks:
                if len(self.pool) < self.pool_cap:
                    self.pool.append(list(res[i].events))
            return res
        if not self.pool:                                  # no calibration pool (should not happen): fall back to a within-batch permutation
            ev = [res[i].events for i in oks]
            perm = self.rng.permutation(len(oks))
            for pos, i in enumerate(oks):
                res[i] = SandboxResult(res[i].obs_id, "ok", ev[perm[pos]], res[i].wall_seconds, res[i].mode, res[i].is_simulated)
            return res
        for i in oks:
            ev = self.pool[int(self.rng.integers(0, len(self.pool)))]
            res[i] = SandboxResult(res[i].obs_id, "ok", list(ev), res[i].wall_seconds, res[i].mode, res[i].is_simulated)
        return res


class UnavailableSandbox(_ControlWrapper):
    """CONTROL / failure injection: after activation EVERY execution fails (sandbox outage)."""

    def execute_many(self, reqs):
        self.n_executed += len(reqs)
        if not self.active:
            return self.inner.execute_many(reqs)
        return [SandboxResult(r.obs_id, "error", [], 0.0, self.mode, self.is_simulated, "sandbox unavailable")
                for r in reqs]


def build_simulated_sandbox(sc, seed: int):
    from .core import make_rng
    inner = (SimulatedSandbox(make_rng(seed, "sandbox"), sc.signal, sc.detectability, sc.benign_noise)
             if sc.model == "v1_label_oracle" else SimulatedBehaviorSandbox(make_rng(seed, "sandbox"), sc))
    if sc.control == "shuffled":
        return ShuffledEvidenceSandbox(inner, make_rng(seed, "shuffle"))
    if sc.control == "global_random":
        return GlobalRandomEvidenceSandbox(inner, make_rng(seed, "global_random"))
    if sc.control == "unavailable":
        return UnavailableSandbox(inner)
    return inner


# ----------------------------------------------------------------------------- recorded
class RecordedTelemetrySandbox(SandboxAdapter):
    """Replays telemetry recorded elsewhere. JSONL lines:
       {"payload_ref": "...", "events": [{"kind": "process_spawn", "count": 3}, ...]}"""
    mode = "RECORDED"
    source = "RECORDED"
    is_simulated = False

    def __init__(self, path: str):
        self.table: Dict[str, List[TelemetryEvent]] = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.table[str(d["payload_ref"])] = [
                        TelemetryEvent(e["kind"], e.get("name", ""), int(e.get("count", 1))) for e in d["events"]]

    def execute(self, req: SandboxRequest) -> SandboxResult:
        t0 = time.perf_counter()
        if req.payload_ref not in self.table:
            return SandboxResult(req.obs_id, "error", [], 0.0, self.mode, False, "no recording for payload_ref")
        return SandboxResult(req.obs_id, "ok", self.table[req.payload_ref], time.perf_counter() - t0, self.mode, False)


# ----------------------------------------------------------------------------- container
@dataclass
class ExecSpec:
    image: str
    argv: List[str]
    timeout_s: int = 30


class ContainerSandboxAdapter(SandboxAdapter):
    """Executes an artefact inside a locked-down Docker container under strace and parses the trace.

    STATUS: UNTESTED against live Docker in this project. Requires a local Docker daemon and an image that
    contains `strace`. Docker is NOT a strong isolation boundary: run real malware only inside a dedicated,
    disposable VM/host. `payload_resolver` maps a request to the artefact to run (e.g. reconstructed file)."""
    mode = "CONTAINER"
    source = "CONTAINER"
    is_simulated = False

    def __init__(self, payload_resolver: Callable[[SandboxRequest], ExecSpec], allow_unavailable: bool = False):
        ok, why = docker_probe()
        if not ok and not allow_unavailable:
            raise EnvironmentNotSupported(f"Docker unavailable here ({why}). Run this adapter locally/in a VM host.")
        self.resolver = payload_resolver

    def build_command(self, spec: ExecSpec, name: str) -> List[str]:
        script = ('strace -f -qq -o /tmp/evcl.trace -- "$@" >/dev/null 2>&1; '
                  'echo ===EVCL_TRACE===; cat /tmp/evcl.trace')
        return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                "--tmpfs", "/tmp:rw,size=64m", "--memory", "256m", "--pids-limit", "128",
                "--cap-drop", "ALL", "--cap-add", "SYS_PTRACE", "--security-opt", "no-new-privileges",
                spec.image, "sh", "-c", script, "evcl", *spec.argv]

    def execute(self, req: SandboxRequest) -> SandboxResult:
        spec = self.resolver(req)
        name = f"evcl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        try:
            r = subprocess.run(self.build_command(spec, name), capture_output=True, text=True,
                               timeout=spec.timeout_s + 10)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return SandboxResult(req.obs_id, "timeout", [], time.perf_counter() - t0, self.mode, False, "timeout")
        wall = time.perf_counter() - t0
        if "===EVCL_TRACE===" not in r.stdout:
            return SandboxResult(req.obs_id, "error", [], wall, self.mode, False, (r.stderr or "no trace")[:300])
        trace = r.stdout.split("===EVCL_TRACE===", 1)[1]
        return SandboxResult(req.obs_id, "ok", parse_strace(trace), wall, self.mode, False)


class VMSandboxAdapter(SandboxAdapter):
    """Interface stub for VM-grade isolation (KVM/Firecracker/CAPE). Not implemented."""
    mode = "VM"
    source = "VM"

    def __init__(self, *a, **k):
        raise EnvironmentNotSupported("VMSandboxAdapter is an interface stub; integrate your VM backend locally.")

    def execute(self, req):  # pragma: no cover
        raise NotImplementedError


class XDPTriageAdapter:
    """Kernel fast-path triage stub. Probes prerequisites and refuses to run otherwise."""

    def __init__(self):
        from .core import xdp_probe
        ok, why = xdp_probe()
        if not ok:
            raise EnvironmentNotSupported("eBPF/XDP fast path unavailable: " + "; ".join(why))
        raise NotImplementedError("Prerequisites look present, but the BPF program/map reader is not part of "
                                  "this notebook. Export the triage threshold/model to a BPF map locally.")
"""evcl.core -- configuration, validation, seeding, environment capture, capability probes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Tuple

import numpy as np


class EnvironmentNotSupported(RuntimeError):
    """A component needs infrastructure this runtime cannot provide. Components NEVER fall back to
    simulation silently; simulation must be chosen explicitly and is always labelled SIMULATED."""


class InvalidExperiment(ValueError):
    """The configuration would produce scientifically invalid or leaky results."""


# ----------------------------------------------------------------------------- seeds / labels
DEV_SEEDS = (0, 1, 2)
FINAL_SEED_MIN = 1000                       # seeds >= 1000 are RESERVED for final runs
FINAL_SEEDS = tuple(range(1000, 1010))
MIN_FINAL_SEEDS = 10
KNOWN_ATTACKS = ("label_poisoning", "clean_label", "replay_flood", "replay_duplicate", "backdoor", "slow_poisoning",
                 "sandbox_evasion", "triage_aware_label")
ALLOCATIONS = ("random", "balanced", "uncertainty", "novelty_balanced")
VERIFICATION_MODES = ("none", "selective", "strict")

PIPELINE_LABEL = {
    "BC": "BC class-balanced throttle (curation control; not a defence)",
    "B0": "B0 frozen / reject-all (trivial)",
    "B2b": "B2b trimmed-loss filter (DEV PROXY)",
    "B1": "B1 no-defence (dev baseline)",
    "B2": "B2 kNN-sanitiser (DEV PROXY; not a published defence)",
    "B3": "B3 execution-verified (proposed, unvalidated)",
    "BR": "BR random-throttle (control; not a defence)",
}
BASELINE_STATUS = {
    "BC": "control: class-balanced random admission at a fixed fraction (isolates data-curation effects from evidence)",
    "B0": "trivial baseline: never learns from the stream (reject-all)",
    "B2b": "DEVELOPMENT PROXY (generic trimmed-loss filter) - not a reproduction of a published defence",
    "B1": "development baseline (no defence)",
    "B2": "DEVELOPMENT PROXY - not a faithful reproduction of any published defence; NOT state-of-the-art",
    "B3": "proposed method (unvalidated)",
    "BR": "control (random admission; isolates the effect of admitting less data)",
}
MODE_LABEL = {"none": "NO verification", "selective": "SELECTIVE verification", "strict": "STRICT verification"}


# ----------------------------------------------------------------------------- configs
@dataclass(frozen=True)
class DataCfg:
    kind: str = "synthetic"             # "synthetic" (pipeline validation ONLY) | "csv" (real flow table)
    n_features: int = 20
    n_classes: int = 5                  # class 0 = benign, 1..n-1 = attack families
    n_chunks: int = 24
    chunk_size: int = 400
    benign_frac: float = 0.7
    bootstrap_per_class: int = 400      # trusted initial training data (base classes only)
    test_per_class: int = 300
    class_sep: float = 3.0
    label_noise_rate: float = 0.0       # natural (non-attacker) label noise on the labelling channel
    calib_per_class: int = 200          # CALIBRATION period (between bootstrap and stream): thresholds/fusion only
    final_test_per_class: int = 300     # FINAL test period (only materialised when run_mode == "final")
    drift_chunk: int = -1               # synthetic: abrupt covariate shift of ALL class means at this chunk (-1 = none)
    drift_norm: float = 0.0
    # (class, first_chunk, last_chunk_exclusive); empty tuple => auto schedule (csv only)
    schedule: Tuple[Tuple[int, int, int], ...] = ((0, 0, 24), (1, 0, 8), (2, 0, 20), (3, 5, 24), (4, 17, 24))
    base_classes: Tuple[int, ...] = (0, 1, 2)
    trigger_features: int = 3
    trigger_sigma: float = 4.0
    csv_path: str = ""
    csv_label_col: str = "label"
    csv_benign_label: str = "BENIGN"
    csv_time_col: str = ""              # numeric/datetime column => chronological split (test = LATEST data)
    csv_allow_random_split: bool = False  # random split is NOT temporally valid; development use only
    csv_entity_col: str = ""            # e.g. src host / session id: entities spanning periods are PURGED
    csv_final_frac: float = 0.15        # latest fraction of each class's timeline reserved as FINAL test period
    csv_calib_frac: float = 0.10        # fraction after the bootstrap period used as CALIBRATION period


@dataclass(frozen=True)
class LearnerCfg:
    algo: str = "softmax_dynamic"       # "softmax_dynamic" (classes discovered from admitted labels) | "sgd_prealloc_v1" (legacy)
    lr: float = 0.05
    batch_size: int = 64
    l2: float = 1e-4
    max_classes: int = 16               # hard bound on label ids the learner will ever accept (labels are attackable)
    epochs: int = 3
    bootstrap_epochs: int = 20
    eta0: float = 0.02
    alpha: float = 1e-4
    replay_capacity: int = 1500
    replay_batch: int = 400
    buffer_policy: str = "reservoir"    # "reservoir" | "class_balanced"
    class_balanced_updates: bool = False  # softmax learner only: inverse-class-frequency sample weights in every SGD batch (strong-baseline option)


@dataclass(frozen=True)
class VerificationCfg:
    """none      : no behavioural verification; everything is admitted (== B1 learner).
       selective : only AMBIGUOUS observations (not confident-normal, or model/label disagreement) are verified;
                   unverified confident-normal fast-path traffic is admitted (+ random audits).
       strict    : EVERY observation must be verified and pass the gate before it can enter memory
                   (no fast path; verification-budget overflow => rejected)."""
    mode: str = "selective"


@dataclass(frozen=True)
class TriageCfg:
    low_threshold: float = 0.2          # used when threshold_source == "fixed"
    threshold_source: str = "calibration"   # "calibration": quantile of P(attack) on benign CALIBRATION data | "fixed"
    benign_pass_quantile: float = 0.90


@dataclass(frozen=True)
class SandboxCfg:
    backend: str = "simulated"          # "simulated" | "external" (adapter object passed in code)
    model: str = "v2_behavior"          # SIMULATED model: "v2_behavior" | "v1_label_oracle" (legacy, near-oracle)
    control: str = "none"               # negative controls: "none" | "shuffled" (evidence permuted across observations)
                                        #                    | "unavailable" (every execution fails)
                                        #                    | "global_random" (telemetry drawn at random from the calibration-period pool)
    budget_per_chunk: int = 120         # max sandbox executions per chunk
    allocation: str = "random"          # how over-budget verification slots are prioritised: random | balanced | uncertainty | novelty_balanced
    detectability: float = 0.5          # SIMULATED: P(env-sensitive evasive sample detects sandbox)
    signal: float = 1.0                 # SIMULATED: 0 = telemetry independent of ground truth (null)
    benign_noise: float = 0.03          # SIMULATED (v1): P(benign behaves partly like malware)
    benign_suspicious: float = 0.05     # SIMULATED (v2): P(benign shows suspicious admin-like behaviour)
    overlap: float = 0.15               # SIMULATED (v2): P(malware is behaviourally dormant/benign-like)
    telemetry_loss: float = 0.15        # SIMULATED (v2): per-event drop probability
    window_frac: float = 0.7            # SIMULATED (v2): fraction of malicious behaviour inside the observation window
    delay_prob: float = 0.2             # SIMULATED (v2): P(activation happens after the window)
    failure_rate: float = 0.05          # SIMULATED (v2): P(execution times out / fails)
    evasion_delay_prob: float = 0.8     # SIMULATED (v2): delayed-activation evasion strength
    mimicry: float = 0.7                # SIMULATED (v2): decoy-event strength for mimicry evasion
    calib_n: int = 600                  # legacy calibration corpus size (calibration_source == "legacy_sim")


@dataclass(frozen=True)
class EvidenceCfg:
    fusion: str = "calibrated"          # "calibrated" | "rules"
    t_benign: float = 0.2               # used when threshold_source == "fixed"
    t_malicious: float = 0.8
    threshold_source: str = "calibration"   # thresholds fit on CALIBRATION-period telemetry only
    target_fpr: float = 0.02            # calibration-period benign with p >= t_malicious
    target_fnr: float = 0.02            # calibration-period malicious with p <= t_benign


@dataclass(frozen=True)
class AdmissionCfg:
    require_label_agreement: bool = True
    relabel_from_evidence: bool = False
    audit_rate: float = 0.05            # selective mode only: random audit of fast-path traffic


@dataclass(frozen=True)
class B2Cfg:
    k: int = 10
    min_agreement: float = 0.5


DEFAULT_ATTACKS: Dict[str, Dict[str, Any]] = {
    "label_poisoning": dict(budget=0.08, start=6, end=20),
    "clean_label": dict(budget=0.08, start=6, end=20, alpha=0.5, target_class=2),
    "replay_flood": dict(start=12, end=15, flood_factor=3.0),
    "replay_duplicate": dict(budget=0.08, start=6, end=20, repeats=10),
    "backdoor": dict(budget=0.08, start=4, end=20),
    "slow_poisoning": dict(budget=0.01, start=0, end=24),
    "sandbox_evasion": dict(budget=0.08, start=6, end=20, evasive_fraction=1.0, evasion_mode="mixed"),
}


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = "evcl_default"
    run_mode: str = "development"       # "development" (seeds < 1000) | "final" (>=10 seeds, all >= 1000)
    seeds: Tuple[int, ...] = DEV_SEEDS
    data: DataCfg = field(default_factory=DataCfg)
    learner: LearnerCfg = field(default_factory=LearnerCfg)
    verification: VerificationCfg = field(default_factory=VerificationCfg)
    triage: TriageCfg = field(default_factory=TriageCfg)
    sandbox: SandboxCfg = field(default_factory=SandboxCfg)
    evidence: EvidenceCfg = field(default_factory=EvidenceCfg)
    admission: AdmissionCfg = field(default_factory=AdmissionCfg)
    b2: B2Cfg = field(default_factory=B2Cfg)
    random_admit_p: float = 0.5         # BR control
    curation_frac: float = 0.127        # BC control: fraction of each chunk admitted, with equal quota per observed label
    trim_frac: float = 0.10             # B2b: fraction of highest-loss known-class samples rejected per chunk
    calibration_source: str = "period"  # "period" (calibration period of the stream) | "legacy_sim" (v1 label-built corpus)
    attacks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_ATTACKS))
    eval_every: int = 1
    checkpoint_every: int = 4
    allow_simulated_sandbox: bool = False


_HASH_EXCLUDE = ("experiment_name", "seeds", "checkpoint_every", "run_mode")


def cfg_to_dict(cfg: RunConfig) -> Dict[str, Any]:
    return json.loads(json.dumps(asdict(cfg), default=str))


def config_hash(cfg: RunConfig) -> str:
    d = cfg_to_dict(cfg)
    for k in _HASH_EXCLUDE:
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _set(obj, parts, value):
    head, rest = parts[0], parts[1:]
    new_child = value
    if rest:
        child = obj[head] if isinstance(obj, dict) else getattr(obj, head)
        new_child = _set(child, rest, value)
    if isinstance(obj, dict):
        d = dict(obj)
        d[head] = new_child
        return d
    return replace(obj, **{head: new_child})


def apply_overrides(cfg: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    for path, value in overrides.items():
        cfg = _set(cfg, path.split("."), value)
    return cfg


STREAM_AFFECTING = ("data.", "attacks.")


def touches_stream(overrides: Dict[str, Any]) -> bool:
    return any(k.startswith(STREAM_AFFECTING) for k in overrides)


# ----------------------------------------------------------------------------- validation
def validate_config(cfg: RunConfig) -> List[str]:
    """Raise InvalidExperiment for configurations that are invalid or would leak; return warnings."""
    err: List[str] = []
    warn: List[str] = []
    d, s, ev, ad, v = cfg.data, cfg.sandbox, cfg.evidence, cfg.admission, cfg.verification
    seeds = list(cfg.seeds)
    if cfg.run_mode not in ("development", "final"):
        err.append(f"run_mode must be 'development' or 'final', got {cfg.run_mode!r}")
    if len(set(seeds)) != len(seeds):
        err.append("duplicate seeds")
    if any((not isinstance(x, (int, np.integer))) or x < 0 for x in seeds):
        err.append("seeds must be non-negative integers")
    if cfg.run_mode == "final":
        if len(seeds) < MIN_FINAL_SEEDS:
            err.append(f"final runs need >= {MIN_FINAL_SEEDS} independent seeds (got {len(seeds)})")
        if any(x < FINAL_SEED_MIN for x in seeds):
            err.append(f"final runs must use only reserved seeds >= {FINAL_SEED_MIN}")
    else:
        if any(x >= FINAL_SEED_MIN for x in seeds):
            err.append(f"seeds >= {FINAL_SEED_MIN} are reserved for final runs; do not use them in development")
        if len(seeds) < MIN_FINAL_SEEDS:
            warn.append(f"development run with {len(seeds)} seeds: CIs/tests are unreliable (final needs >= {MIN_FINAL_SEEDS})")
    lr_, tr_, sb_ = cfg.learner, cfg.triage, cfg.sandbox
    if lr_.algo not in ("softmax_dynamic", "mlp_dynamic", "forest_refit", "sgd_prealloc_v1"):
        err.append("learner.algo must be 'softmax_dynamic', 'mlp_dynamic', 'forest_refit' or 'sgd_prealloc_v1'")
    if lr_.class_balanced_updates and lr_.algo == "sgd_prealloc_v1":
        err.append("learner.class_balanced_updates requires a dynamic learner (not 'sgd_prealloc_v1')")
    if lr_.max_classes < d.n_classes:
        err.append("learner.max_classes must be >= data.n_classes")
    if tr_.threshold_source not in ("calibration", "fixed") or ev.threshold_source not in ("calibration", "fixed"):
        err.append("threshold_source must be 'calibration' or 'fixed'")
    if not (0.5 <= tr_.benign_pass_quantile < 1):
        err.append("triage.benign_pass_quantile must be in [0.5,1)")
    if not (0 < ev.target_fpr < 0.5 and 0 < ev.target_fnr < 0.5):
        err.append("evidence.target_fpr/target_fnr must be in (0,0.5)")
    if sb_.model not in ("v2_behavior", "v1_label_oracle") or sb_.control not in ("none", "shuffled", "unavailable", "global_random"):
        err.append("invalid sandbox.model / sandbox.control")
    for name, val in (("sandbox.overlap", sb_.overlap), ("sandbox.benign_suspicious", sb_.benign_suspicious),
                      ("sandbox.telemetry_loss", sb_.telemetry_loss), ("sandbox.window_frac", sb_.window_frac),
                      ("sandbox.delay_prob", sb_.delay_prob), ("sandbox.failure_rate", sb_.failure_rate),
                      ("sandbox.evasion_delay_prob", sb_.evasion_delay_prob), ("sandbox.mimicry", sb_.mimicry)):
        if not (0 <= val <= 1):
            err.append(f"{name} must be in [0,1]")
    if sb_.allocation not in ALLOCATIONS:
        err.append(f"sandbox.allocation must be one of {ALLOCATIONS}")
    if not (0 < cfg.curation_frac <= 1):
        err.append("curation_frac must be in (0,1]")
    if sb_.window_frac <= 0:
        err.append("sandbox.window_frac must be > 0")
    if cfg.calibration_source not in ("period", "legacy_sim"):
        err.append("calibration_source must be 'period' or 'legacy_sim'")
    if not (0 <= cfg.trim_frac < 0.5):
        err.append("trim_frac must be in [0,0.5)")
    if d.calib_per_class < 20 or d.final_test_per_class < 1:
        err.append("calib_per_class >= 20 and final_test_per_class >= 1 required")
    if d.drift_norm < 0 or (d.drift_chunk >= d.n_chunks) or (d.drift_chunk >= 0 and d.drift_norm == 0):
        err.append("drift_chunk must be < n_chunks and (if enabled) drift_norm > 0")
    if d.drift_chunk >= 0 and d.kind != "synthetic":
        err.append("drift injection is synthetic-only; real data drift is whatever the dataset contains")
    if v.mode not in VERIFICATION_MODES:
        err.append(f"verification.mode must be one of {VERIFICATION_MODES}")
    if v.mode == "strict" and ad.audit_rate > 0:
        err.append("strict mode verifies everything; admission.audit_rate must be 0 (ambiguous config)")
    if v.mode != "none" and s.backend == "simulated" and not cfg.allow_simulated_sandbox:
        err.append("simulated sandbox telemetry requires allow_simulated_sandbox=True (explicit acknowledgement)")
    if s.backend not in ("simulated", "external"):
        err.append("sandbox.backend must be 'simulated' or 'external'")
    if not (0 <= ev.t_benign < ev.t_malicious <= 1):
        err.append("need 0 <= evidence.t_benign < evidence.t_malicious <= 1")
    if not (0 <= cfg.triage.low_threshold <= 1):
        err.append("triage.low_threshold must be in [0,1]")
    for name, val in (("admission.audit_rate", ad.audit_rate), ("sandbox.detectability", s.detectability),
                      ("sandbox.signal", s.signal), ("sandbox.benign_noise", s.benign_noise),
                      ("b2.min_agreement", cfg.b2.min_agreement), ("random_admit_p", cfg.random_admit_p)):
        if not (0 <= val <= 1):
            err.append(f"{name} must be in [0,1]")
    if not (0 <= d.label_noise_rate <= 0.5):
        err.append("data.label_noise_rate must be in [0,0.5]")
    if s.budget_per_chunk < 0 or s.calib_n < 20:
        err.append("sandbox.budget_per_chunk >= 0 and sandbox.calib_n >= 20 required")
    if cfg.eval_every < 1 or cfg.checkpoint_every < 1:
        err.append("eval_every and checkpoint_every must be >= 1")
    if cfg.learner.replay_capacity < 1 or cfg.b2.k < 1:
        err.append("replay_capacity and b2.k must be >= 1")
    if d.chunk_size < 10 or d.n_chunks < 4 or d.bootstrap_per_class < 1 or d.test_per_class < 1:
        err.append("degenerate data sizes")
    # ---- schedule / temporal validity
    sched = d.schedule
    if d.kind == "synthetic" and not sched:
        err.append("synthetic data requires an explicit schedule")
    if d.kind not in ("synthetic", "csv"):
        err.append("data.kind must be 'synthetic' or 'csv'")
    if sched:
        first: Dict[int, int] = {}
        for (c, st, en) in sched:
            if not (0 <= c < d.n_classes):
                err.append(f"schedule class {c} outside [0,{d.n_classes})")
            if not (0 <= st < en <= d.n_chunks):
                err.append(f"schedule window ({c},{st},{en}) invalid for n_chunks={d.n_chunks}")
            first[c] = min(first.get(c, 10 ** 9), st)
        if 0 not in first:
            err.append("benign class 0 must appear in the schedule")
        if 0 in first and not any(c == 0 and st == 0 and en == d.n_chunks for (c, st, en) in sched):
            warn.append("benign class is not active in every chunk")
        if 0 not in d.base_classes or len(d.base_classes) < 2:
            err.append("base_classes must contain benign class 0 and at least one attack class")
        for c in d.base_classes:
            if c in first and first[c] > 0:
                err.append(f"base class {c} first appears at chunk {first[c]} but is in the trusted bootstrap set: "
                           "FUTURE-CLASS LEAKAGE into earlier tasks")
            if c not in first:
                err.append(f"base class {c} is not in the schedule")
    # ---- data source
    if d.kind == "csv":
        if not d.csv_path or not os.path.exists(d.csv_path):
            err.append(f"csv_path not found: {d.csv_path!r}")
        if not d.csv_time_col and not d.csv_allow_random_split:
            err.append("csv data without csv_time_col requires csv_allow_random_split=True "
                       "(random splits are NOT temporally valid; near-duplicate flows leak across splits)")
        if not d.csv_time_col and d.csv_allow_random_split:
            warn.append("random CSV split: temporal validity NOT established")
    else:
        warn.append("synthetic data: pipeline validation only, not evidence about real NIDS performance")
    if d.label_noise_rate == 0:
        warn.append("labelling channel is perfect for non-attacker observations (unrealistic; see data.label_noise_rate)")
    # ---- attacks
    for name, p in cfg.attacks.items():
        if name not in KNOWN_ATTACKS:
            err.append(f"unknown attack {name!r}")
            continue
        st, en = int(p.get("start", 0)), int(p.get("end", d.n_chunks))
        if not (0 <= st < en):
            err.append(f"attack {name}: invalid window ({st},{en})")
        if st >= d.n_chunks:
            err.append(f"attack {name}: window starts at {st} >= n_chunks={d.n_chunks}; it would inject nothing")
        if en > d.n_chunks:
            warn.append(f"attack {name}: window end {en} > n_chunks; truncated")
        if "budget" in p and not (0 < p["budget"] <= 1):
            err.append(f"attack {name}: budget must be in (0,1]")
        if name == "replay_flood" and p.get("flood_factor", 0) <= 0:
            err.append("replay_flood.flood_factor must be > 0")
        if name == "clean_label" and not (1 <= int(p.get("target_class", 0)) < d.n_classes):
            err.append("clean_label.target_class must be an attack class id")
        if name == "replay_duplicate" and int(p.get("repeats", 0)) < 2:
            err.append("replay_duplicate.repeats must be >= 2")
        if name == "sandbox_evasion" and p.get("evasion_mode", "mixed") not in ("env", "delayed", "mimicry", "partial", "mixed"):
            err.append("sandbox_evasion.evasion_mode invalid")
    if err:
        raise InvalidExperiment("; ".join(err))
    return warn


def legacy_config(**kw) -> "RunConfig":
    """Reproduces the v1 (Colab-executed) experimental setting: pre-allocated SGD learner, near-oracle simulator,
    fixed thresholds, label-built calibration corpus, no drift, v1 schedule/attack windows. Regression use only."""
    base = RunConfig(
        data=DataCfg(schedule=((0, 0, 24), (1, 0, 6), (2, 0, 12), (3, 8, 24), (4, 15, 24))),
        learner=LearnerCfg(algo="sgd_prealloc_v1"),
        triage=TriageCfg(threshold_source="fixed"),
        sandbox=SandboxCfg(model="v1_label_oracle"),
        evidence=EvidenceCfg(threshold_source="fixed"),
        calibration_source="legacy_sim",
        attacks={"label_poisoning": dict(budget=0.08, start=6, end=18),
                 "clean_label": dict(budget=0.08, start=6, end=18, alpha=0.5, target_class=2),
                 "replay_flood": dict(start=10, end=13, flood_factor=3.0),
                 "backdoor": dict(budget=0.08, start=4, end=18),
                 "slow_poisoning": dict(budget=0.01, start=0, end=24),
                 "sandbox_evasion": dict(budget=0.08, start=6, end=18, evasive_fraction=1.0)},
        allow_simulated_sandbox=True)
    return apply_overrides(base, kw) if kw else base


# ----------------------------------------------------------------------------- seeding
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)   # only affects child processes


def make_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(name.encode())]))


# ----------------------------------------------------------------------------- environment
def source_hash(pkg_dir: str | None = None) -> str:
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in sorted(os.listdir(pkg_dir)):
        if fn.endswith(".py"):
            with open(os.path.join(pkg_dir, fn), "rb") as f:
                h.update(fn.encode())
                h.update(f.read())
    return h.hexdigest()[:16]


def capture_environment() -> Dict[str, Any]:
    import importlib.metadata as md
    pk = {}
    for p in ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "psutil"):
        try:
            pk[p] = md.version(p)
        except Exception:
            pk[p] = None
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                cpu_count=os.cpu_count(), packages=pk, in_colab="google.colab" in sys.modules,
                code_hash=source_hash())


def docker_probe() -> Tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker CLI not found"
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=8)
        return (r.returncode == 0, "ok" if r.returncode == 0 else "docker daemon not reachable")
    except Exception as e:  # noqa: BLE001
        return False, f"docker probe failed: {e}"


def xdp_probe() -> Tuple[bool, List[str]]:
    reasons = []
    if platform.system() != "Linux":
        reasons.append("not Linux")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        reasons.append("not root")
    if not os.path.exists("/sys/kernel/btf/vmlinux"):
        reasons.append("no kernel BTF (/sys/kernel/btf/vmlinux)")
    try:
        import bcc  # noqa: F401
        has_bcc = True
    except Exception:
        has_bcc = False
    if not (shutil.which("bpftool") or has_bcc):
        reasons.append("neither bpftool nor bcc found")
    if "google.colab" in sys.modules:
        reasons.append("hosted Colab VM: no NIC/XDP attach point under experimenter control")
    return (len(reasons) == 0, reasons)


def capability_report() -> List[Dict[str, str]]:
    ok_x, why_x = xdp_probe()
    ok_d, why_d = docker_probe()
    return [
        dict(component="Triage: userspace model score", status="RUNS (measured userspace compute only)",
             detail="NOT a kernel data plane; latency/throughput here are NOT eBPF/XDP numbers."),
        dict(component="eBPF/XDP fast path", status="PROBE OK, ADAPTER IS A STUB (never executed)" if ok_x else "UNAVAILABLE",
             detail="; ".join(why_x) if why_x else "prerequisites present; BPF program/map reader not implemented"),
        dict(component="Sandbox telemetry: SIMULATED", status="RUNS (SIMULATED - modeled, not measured)",
             detail="Generated from ground truth + assumed noise. Never present as sandbox measurements."),
        dict(component="Sandbox: Docker container + strace", status="UNTESTED, runnable only locally" if ok_d else "UNAVAILABLE",
             detail=why_d + "; Docker is NOT a strong isolation boundary for real malware."),
        dict(component="Sandbox: recorded-telemetry import", status="RUNS (needs your measured JSONL)",
             detail="Separate from SIMULATED; results are tagged RECORDED."),
        dict(component="Sandbox: VM (KVM/Firecracker/CAPE)", status="UNAVAILABLE (interface stub, never executed)",
             detail="No VM measurements exist in this notebook."),
    ]
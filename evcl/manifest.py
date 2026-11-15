"""evcl.manifest -- provenance manifest: environment, dependencies, configuration hashes, data fingerprints, seed assignments,
timestamps and file hashes. Generated from files on disk (never typed by hand)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

from .core import capture_environment, source_hash


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _freeze():
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60).stdout
        keep = ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "psutil", "nbformat", "ipykernel")
        return [l for l in out.splitlines() if l.split("==")[0].lower() in keep]
    except Exception as e:  # noqa: BLE001
        return [f"pip freeze failed: {e}"]


def build_manifest(root: str, extra_deviations=None) -> dict:
    import pandas as pd
    runs = os.path.join(root, "runs")
    m = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), environment=capture_environment(),
             pip_freeze_selected=_freeze(), code_hash=source_hash(os.path.join(root, "evcl")),
             module_sha256={os.path.basename(p): _sha(p) for p in sorted(glob.glob(os.path.join(root, "evcl", "*.py")))},
             seed_plan=dict(development_exploratory=[0, 1, 2, 3, 4], final_confirmatory=list(range(1000, 1010)),
                            rule="seeds >= 1000 reserved for final runs; registry forbids reuse across modes and forbids evaluating a final seed under a second configuration"),
             data_sources=dict(real_dataset="NONE (no timestamped real dataset accessible; real-data experiment pending)",
                               synthetic_generator="evcl.data.build_stream (seeded; fingerprints per trial below)"),
             telemetry="SIMULATED (evcl.sandbox.SimulatedBehaviorSandbox); no measured, eBPF/XDP, VM or Docker telemetry exists in these results",
             deviations=list(extra_deviations or []), stages={})
    reg = os.path.join(runs, "registry", "evcl_validation", "seed_registry.json")
    if os.path.exists(reg):
        m["seed_registry"] = json.load(open(reg))
    for f in ("primary_doses.json", "protocol_frozen.json", "dev_attack_dose.csv"):
        p = os.path.join(runs, f)
        if os.path.exists(p):
            m.setdefault("input_files", {})[f] = dict(sha256=_sha(p), mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))))
    for stage in ("dev", "final", "smoke"):
        base = os.path.join(runs, stage, "evcl_validation")
        if not os.path.isdir(base):
            continue
        for h in sorted(os.listdir(base)):
            d = os.path.join(base, h)
            s = dict(config_hash=h, n_trial_files=len(glob.glob(os.path.join(d, "trials", "*.json"))))
            for fn in ("config.json", "env.json", "results_main.csv", "results_exploratory_sweeps.csv"):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    s.setdefault("files", {})[fn] = dict(sha256=_sha(p), mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))))
            p = os.path.join(d, "results_main.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                s["seeds"] = sorted(int(x) for x in df.seed.unique())
                s["run_mode"] = sorted(df.run_mode.unique().tolist())
                s["stage_label"] = sorted(df.stage.unique().tolist())
                s["code_hash_in_env_json"] = json.load(open(os.path.join(d, "env.json")))["code_hash"]
                s["stream_fingerprints"] = {f"{a}|seed{sd}": fp for (a, sd), fp in df.groupby(["attack", "seed"]).stream_fingerprint.first().items()}
                s["eval_fingerprints"] = {f"seed{sd}": fp for sd, fp in df.groupby("seed").eval_fingerprint.first().items()}
                s["telemetry_sources"] = sorted(df.telemetry_source.unique().tolist())
                s["headline_split"] = sorted(df.headline_split.unique().tolist())
            m["stages"][stage] = s
    logs = os.path.join(root, "logs")
    if os.path.isdir(logs):
        m["log_files"] = {os.path.basename(p): _sha(p) for p in sorted(glob.glob(os.path.join(logs, "*"))) if os.path.isfile(p)}
    res = os.path.join(root, "results")
    if os.path.isdir(res):
        m["result_files"] = {os.path.relpath(p, root): _sha(p) for p in sorted(glob.glob(os.path.join(res, "**", "*"), recursive=True)) if os.path.isfile(p)}
    return m
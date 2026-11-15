"""evcl.experiment -- scenario construction, leakage-safe trial runner (checkpoint/resume), grids, protocol freeze."""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .core import (BASELINE_STATUS, RunConfig, apply_overrides, config_hash, make_rng, set_global_seed,
                   touches_stream, validate_config, cfg_to_dict)
from .data import StreamData, build_stream
from .evaluation import (evaluate_final, evaluate_model, finalize_metrics, new_accumulator, update_accumulator)
from .pipelines import make_pipeline
from .poisoning import make_attack

SCENARIOS = ["none", "label_poisoning", "clean_label", "replay_flood", "replay_duplicate", "backdoor",
             "slow_poisoning", "sandbox_evasion"]

# ---- pre-declared telemetry-fidelity levels (SIMULATED; applied as dotted overrides) ------------------------
FIDELITY = {
    "HIGH":   {"sandbox.overlap": 0.0, "sandbox.benign_suspicious": 0.0, "sandbox.telemetry_loss": 0.0,
               "sandbox.window_frac": 1.0, "sandbox.delay_prob": 0.0, "sandbox.failure_rate": 0.0},
    "MEDIUM": {},      # defaults in SandboxCfg (chosen a priori; arbitrary, not calibrated to any real sandbox)
    "LOW":    {"sandbox.overlap": 0.40, "sandbox.benign_suspicious": 0.15, "sandbox.telemetry_loss": 0.40,
               "sandbox.window_frac": 0.4, "sandbox.delay_prob": 0.5, "sandbox.failure_rate": 0.20},
}
SEL = {"verification.mode": "selective"}
STR = {"verification.mode": "strict", "admission.audit_rate": 0.0}
MODE_VARIANTS: Dict[str, Dict] = {
    "NONE": {"verification.mode": "none"},
    "SELECTIVE": dict(SEL),
    "STRICT": dict(STR),
}
CONTROL_VARIANTS: Dict[str, Dict] = {
    "SELECTIVE|shuffled-evidence": {**SEL, "sandbox.control": "shuffled"},
    "STRICT|shuffled-evidence": {**STR, "sandbox.control": "shuffled"},
    "SELECTIVE|sandbox-outage": {**SEL, "sandbox.control": "unavailable"},
    "STRICT|sandbox-outage": {**STR, "sandbox.control": "unavailable"},
    "SELECTIVE|null-signal": {**SEL, "sandbox.signal": 0.0},
    "STRICT|null-signal": {**STR, "sandbox.signal": 0.0},
}


def build_scenario(cfg: RunConfig, seed: int, attack: str) -> StreamData:
    base = build_stream(cfg.data, seed, include_final=(cfg.run_mode == "final"))
    if attack == "none":
        return base
    return make_attack(attack, cfg.attacks[attack]).apply(base, make_rng(seed, f"attack:{attack}"))


def trial_key(variant, pipeline, attack, seed):
    return f"{variant}__{pipeline}__{attack}__seed{seed}"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_trial(cfg: RunConfig, pipeline: str, stream: StreamData, seed: int, attack: str, variant="base",
              ckpt=None, stop_after_chunk: Optional[int] = None, sandbox=None, calibration=None):
    """Returns the result dict, or None if stopped early via stop_after_chunk (state is checkpointed)."""
    key = trial_key(variant, pipeline, attack, seed)
    if ckpt is not None and ckpt.has_trial(key):
        return ckpt.load_trial(key)
    set_global_seed(seed)
    fp = stream.fingerprint()
    chash = config_hash(cfg)
    state = ckpt.load_state(key) if ckpt is not None else None
    if state:
        if state["fp"] != fp or state.get("chash") != chash:
            raise RuntimeError(f"{key}: checkpoint was made on a different stream/configuration (code, data or config "
                               "changed) - delete the stale state or use a new run directory")
        pipe, acc, t0 = state["pipe"], state["acc"], state["next_chunk"]
    else:
        view = stream.defender_view()                       # defenders get NO test data, NO meta, NO ground truth
        pipe = make_pipeline(pipeline, cfg, view, seed, sandbox, calibration)
        pipe.setup(view)
        acc, t0 = new_accumulator(), 0
    n = len(stream.chunks)
    for t in range(t0, n):
        chunk = stream.chunks[t]                            # ground truth: runner-side only
        obs = chunk.observed(t)                             # defender-side view
        oracle = chunk.oracle(t) if pipe.needs_oracle else None
        c0, w0 = time.process_time(), time.perf_counter()
        out = pipe.process_chunk(obs, t, oracle)
        update_accumulator(acc, chunk, obs, out, time.perf_counter() - w0, time.process_time() - c0)
        if (t + 1) % cfg.eval_every == 0 or t == n - 1:
            acc["evals"].append(evaluate_model(pipe.learner, stream, t))
        done = t + 1
        stopping = stop_after_chunk is not None and done >= stop_after_chunk and done < n
        if ckpt is not None and done < n and (done % cfg.checkpoint_every == 0 or stopping):
            ckpt.save_state(key, dict(pipe=pipe, acc=acc, next_chunk=done, fp=fp, chash=chash))
        if stopping:
            return None
    final_eval = evaluate_final(pipe.learner, stream) if cfg.run_mode == "final" else None   # touched ONCE, at the end
    res = finalize_metrics(acc, stream.meta, pipe.learner, n, pipe.telemetry_source, final_eval,
                           budget=cfg.sandbox.budget_per_chunk if pipeline == "B3" else None)
    if attack != "none" and res["poison_total"] == 0:
        warnings.warn(f"{key}: attack '{attack}' injected 0 poisoned observations (window outside stream?).")
    res.update(variant=variant, pipeline=pipeline, attack=attack, seed=seed, key=key, stream_fingerprint=fp,
               eval_fingerprint=stream.eval_fingerprint(), config_hash=chash,
               stage="confirmatory" if cfg.run_mode == "final" else "exploratory",
               telemetry_source=pipe.telemetry_source, triage_backend=pipe.triage_backend,
               baseline_status=BASELINE_STATUS[pipeline],
               verification_mode=cfg.verification.mode if pipeline == "B3" else "n/a",
               data_kind=stream.meta["dataset"], split_type=stream.meta["split_type"],
               temporal_split=bool(stream.meta["temporal_split"]), label_noise_rate=cfg.data.label_noise_rate,
               drift_chunk=cfg.data.drift_chunk, run_mode=cfg.run_mode, learner_algo=cfg.learner.algo,
               dataset_sha256=_sha256_file(cfg.data.csv_path) if cfg.data.kind == "csv" else "synthetic-generator",
               config_label_channel="ground truth except attacker flips"
               + (f" + {cfg.data.label_noise_rate:.0%} natural noise" if cfg.data.label_noise_rate else ""),
               evals=acc["evals"], **{f"prov_{k}": v for k, v in pipe.provenance().items()})
    if ckpt is not None:
        ckpt.save_trial(key, res)
        ckpt.drop_state(key)
    return res


def run_grid(cfg: RunConfig, ckpt=None, pipelines: Sequence[str] = ("B0", "B1", "BR", "B2", "B2b", "B3"),
             attacks: Sequence[str] = SCENARIOS, variants: Optional[Dict[str, Dict]] = None,
             seeds: Optional[Iterable[int]] = None, verbose=True, sandbox=None, calibration=None,
             registry=None, allow_stream_overrides=False):
    """Every pipeline/variant in a (attack, seed) cell sees the SAME attacked stream (fingerprint recorded)
    unless a variant overrides data./attacks. (refused by default)."""
    variants = variants or {"base": {}}
    for vname, ov in variants.items():
        if touches_stream(ov) and not allow_stream_overrides:
            raise ValueError(f"variant {vname!r} overrides stream-defining settings {list(ov)}; refused because "
                             "variants must be compared on identical streams")
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    vcfgs = {v: apply_overrides(cfg, ov) for v, ov in variants.items()}
    for vcfg in vcfgs.values():
        validate_config(vcfg)
    if registry is not None:
        registry.register(cfg.run_mode, seeds, config_hash(cfg))
    results, done = [], 0
    todo = len(variants) * len(attacks) * len(seeds) * len(pipelines)
    t_start = time.time()
    for seed in seeds:
        for attack in attacks:
            streams: Dict = {}
            for vname, vcfg in vcfgs.items():
                for p in pipelines:
                    key = trial_key(vname, p, attack, seed)
                    if ckpt is not None and ckpt.has_trial(key):
                        results.append(ckpt.load_trial(key))
                    else:
                        sk = vname if touches_stream(variants[vname]) else "_shared"
                        if sk not in streams:
                            streams[sk] = build_scenario(vcfg, seed, attack)       # variant-specific stream only if it changes the stream
                        results.append(run_trial(vcfg, p, streams[sk], seed, attack, vname, ckpt,
                                                 sandbox=sandbox if p == "B3" else None,
                                                 calibration=calibration if p == "B3" else None))
                    done += 1
            if verbose and (done % max(1, len(variants) * len(pipelines) * len(attacks)) == 0):
                print(f"[{done}/{todo}] seed={seed} done  elapsed={time.time()-t_start:.0f}s", flush=True)
    return results


def results_to_df(results):
    import pandas as pd
    return pd.DataFrame([{k: v for k, v in r.items() if k != "evals"} for r in results])


# ----------------------------------------------------------------------------- protocol freeze / manifest
def freeze_protocol(path: str, cfg: RunConfig, design: dict, seeds, hypotheses: list, extra: dict | None = None):
    """Write the confirmatory protocol BEFORE any final run. Refuses to overwrite a different frozen protocol."""
    body = dict(config=cfg_to_dict(cfg), config_hash=config_hash(cfg), design=design, seeds=list(map(int, seeds)),
                hypotheses=hypotheses, extra=extra or {})
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    if os.path.exists(path):
        old = json.load(open(path))
        if old["protocol_sha256"] != digest:
            raise RuntimeError("a DIFFERENT protocol was already frozen at this path; refusing to overwrite")
        return old
    out = dict(body, protocol_sha256=digest, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(out, open(path, "w"), indent=2, default=str)
    return out


def check_protocol(path: str, cfg: RunConfig, design: dict, seeds, hypotheses: list):
    """Final runs must match the frozen protocol exactly."""
    old = json.load(open(path))
    body = dict(config=cfg_to_dict(cfg), config_hash=config_hash(cfg), design=design, seeds=list(map(int, seeds)),
                hypotheses=hypotheses, extra=old.get("extra", {}))
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    if digest != old["protocol_sha256"]:
        raise RuntimeError("configuration/design/seeds/hypotheses differ from the frozen protocol; final run refused")
    return True
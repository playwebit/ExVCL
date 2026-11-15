"""evcl.stages -- experiment definitions shared by scripts/ and the Colab notebook (one code path for dev, final, smoke)."""
from __future__ import annotations

import json
import os
import time

from . import experiment as E
from .audit import assert_results_valid, banner
from .checkpoint import CheckpointManager, SeedRegistry
from .core import (DEFAULT_ATTACKS, DEV_SEEDS, FINAL_SEEDS, DataCfg, RunConfig, apply_overrides, capture_environment,
                   cfg_to_dict, config_hash, validate_config)
from .evaluation import add_pi

DEV_SEEDS_V2 = (0, 1, 2, 3, 4)
BASE_PIPELINES = ("B0", "B1", "BR", "B2", "B2b")
OP_VARIANTS = {  # operating-point sensitivity of the calibrated evidence thresholds (secondary, descriptive)
    **{f"SELECTIVE|target-{int(t*100)}%": {**E.SEL, "evidence.target_fpr": t, "evidence.target_fnr": t} for t in (0.05, 0.10)},
    **{f"STRICT|target-{int(t*100)}%": {**E.STR, "evidence.target_fpr": t, "evidence.target_fnr": t} for t in (0.05, 0.10)}}
SENS_VARIANTS = {   # pre-declared SECONDARY sensitivity analyses (telemetry fidelity, verification budget) - descriptive only
    **{f"{m}|fidelity-{l}": {**base, **E.FIDELITY[l]} for m, base in (("SELECTIVE", E.SEL), ("STRICT", E.STR)) for l in ("LOW", "HIGH")},
    **{f"{m}|budget-{('unlimited' if b == 10 ** 6 else b)}": {**base, "sandbox.budget_per_chunk": b}
       for m, base in (("SELECTIVE", E.SEL), ("STRICT", E.STR)) for b in (30, 240, 10 ** 6)}}
B3_VARIANTS = {**E.MODE_VARIANTS, **E.CONTROL_VARIANTS, **OP_VARIANTS, **SENS_VARIANTS}
SMOKE_ATTACKS = ("none", "label_poisoning", "backdoor")

DESIGN = dict(
    stream="synthetic; drift at chunk 11 (norm 2.0); novel classes at chunks 5 and 17; class 1 leaves at 8, class 2 at 20",
    telemetry="SIMULATED v2_behavior, MEDIUM fidelity (SandboxCfg defaults)", baseline_pipelines=list(BASE_PIPELINES),
    b3_variants=B3_VARIANTS, attacks=E.SCENARIOS, budget_per_chunk=120, thresholds="calibration-period fitted (2%/2%)",
    secondary_sensitivities="fidelity HIGH/LOW, budget 30/240/unlimited, threshold targets 5%/10% (descriptive, pre-declared)")
HYPOTHESES = [
    "P1: seed-level mean ATTACKED headline macro-F1 (mean over the 7 attacks) differs between B3-SELECTIVE and B3-NONE (accept-all); paired over seeds; exact two-sided Wilcoxon.",
    "P2: same for B3-STRICT vs B3-NONE.",
    "P3: B3-SELECTIVE vs B3-SELECTIVE|shuffled-evidence on the same outcome (does real evidence beat uninformative evidence?).",
    "P4: B3-STRICT vs B3-STRICT|shuffled-evidence on the same outcome.",
    "Multiplicity: Holm over P1-P4. Poisoning impact PI, youden, FRR, ASR, per-attack contrasts, controls, sweeps are secondary/descriptive with no significance claim.",
    "Directional expectation stated in advance: P1/P2 positive only if verification protects utility under attack more than it costs in clean-data rejection; the result may be negative and will be reported either way.",
    "Inference is over independent seeds of ONE synthetic generator with SIMULATED telemetry; it does not generalise to real traffic or sandboxes."]


def make_cfg(root: str, run_mode: str, seeds, smoke: bool = False) -> RunConfig:
    doses = json.load(open(os.path.join(root, "runs", "primary_doses.json")))
    attacks = {k: dict(v) for k, v in DEFAULT_ATTACKS.items()}
    for atk, d in doses.items():
        attacks[atk][d["param"]] = d["dose"]
    cfg = RunConfig(experiment_name="evcl_validation", run_mode=run_mode, seeds=tuple(seeds), attacks=attacks,
                    allow_simulated_sandbox=True, data=DataCfg(drift_chunk=11, drift_norm=2.0))
    if smoke:   # small, fast configuration for a smoke test (NOT a scientific configuration)
        cfg = apply_overrides(cfg, {
            "data.n_chunks": 10, "data.chunk_size": 200, "data.bootstrap_per_class": 150, "data.test_per_class": 100,
            "data.calib_per_class": 60, "data.final_test_per_class": 100, "data.drift_chunk": 5,
            "data.schedule": ((0, 0, 10), (1, 0, 4), (2, 0, 9), (3, 3, 10), (4, 7, 10)),
            "attacks.label_poisoning.start": 2, "attacks.label_poisoning.end": 8,
            "attacks.clean_label.start": 2, "attacks.clean_label.end": 8, "attacks.replay_flood.start": 4,
            "attacks.replay_flood.end": 6, "attacks.replay_duplicate.start": 2, "attacks.replay_duplicate.end": 8,
            "attacks.backdoor.start": 2, "attacks.backdoor.end": 8, "attacks.slow_poisoning.start": 0,
            "attacks.slow_poisoning.end": 10, "attacks.sandbox_evasion.start": 2, "attacks.sandbox_evasion.end": 8,
            "learner.replay_capacity": 600, "learner.replay_batch": 200, "sandbox.budget_per_chunk": 60})
    return cfg


def exploratory_sweeps():
    """Development-only exploratory sweeps (not part of the confirmatory design)."""
    sw = {}
    for b in (60,):
        sw[f"SELECTIVE|budget-{b}"] = {**E.SEL, "sandbox.budget_per_chunk": b}
        sw[f"STRICT|budget-{b}"] = {**E.STR, "sandbox.budget_per_chunk": b}
    for knob, val in (("telemetry_loss", 0.5), ("window_frac", 0.3), ("delay_prob", 0.6), ("overlap", 0.4), ("failure_rate", 0.4)):
        sw[f"SELECTIVE|{knob}={val}"] = {**E.SEL, f"sandbox.{knob}": val}
    sw["SELECTIVE|fixed-thresholds"] = {**E.SEL, "triage.threshold_source": "fixed", "evidence.threshold_source": "fixed"}
    sw["SELECTIVE|rule-fusion"] = {**E.SEL, "evidence.fusion": "rules"}
    sw["SELECTIVE|audit0"] = {**E.SEL, "admission.audit_rate": 0.0}
    sw["SELECTIVE|audit20"] = {**E.SEL, "admission.audit_rate": 0.20}
    return sw


def run_stage(root: str, stage: str, smoke: bool = False, verbose: bool = True):
    """stage in {'dev','final','smoke'}. Returns the run directory."""
    assert stage in ("dev", "final", "smoke")
    seeds = {"dev": DEV_SEEDS_V2, "final": FINAL_SEEDS, "smoke": (0, 1)}[stage]
    cfg = make_cfg(root, "final" if stage == "final" else "development", seeds, smoke=(stage == "smoke"))
    for w in validate_config(cfg):
        print("WARNING:", w)
    rdir = os.path.join(root, "runs")
    registry = None if stage == "smoke" else SeedRegistry(os.path.join(rdir, "registry"), "evcl_validation")
    ck = CheckpointManager(os.path.join(rdir, stage), "evcl_validation", config_hash(cfg))
    ck.write_json("config.json", cfg_to_dict(cfg))
    ck.write_json("env.json", capture_environment())
    if stage == "final":
        E.check_protocol(os.path.join(rdir, "protocol_frozen.json"), cfg, DESIGN, cfg.seeds, HYPOTHESES)
        print("final run matches the frozen protocol", flush=True)
    attacks = SMOKE_ATTACKS if stage == "smoke" else E.SCENARIOS
    pipes = ("B0", "B1", "B2") if stage == "smoke" else BASE_PIPELINES
    b3v = {k: B3_VARIANTS[k] for k in ("NONE", "SELECTIVE", "STRICT", "SELECTIVE|shuffled-evidence")} if stage == "smoke" else B3_VARIANTS
    t0 = time.time()
    res = E.run_grid(cfg, ck, pipelines=pipes, attacks=attacks, registry=registry, verbose=False)
    res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=attacks, variants=b3v, registry=registry, verbose=False)
    main = add_pi(E.results_to_df(res))
    assert_results_valid(main)
    main.to_csv(os.path.join(ck.run_dir, "results_main.csv"), index=False)
    print(f"main grid validated: {len(main)} trials in {time.time()-t0:.0f}s | {banner(main)}", flush=True)
    if stage == "dev":
        ex = E.run_grid(cfg, ck, pipelines=("B3",), variants=exploratory_sweeps(), registry=registry, verbose=False)
        ev = {f"evasion-{m}|{nm}": {**ov, "attacks.sandbox_evasion.evasion_mode": m} for m in ("env", "delayed", "mimicry", "partial")
              for nm, ov in (("NONE", E.MODE_VARIANTS["NONE"]), ("SELECTIVE", E.SEL), ("STRICT", E.STR))}
        ex += E.run_grid(cfg, ck, pipelines=("B3",), attacks=("none", "sandbox_evasion"), variants=ev, registry=registry,
                         verbose=False, allow_stream_overrides=True)
        ln = {f"labelnoise{r}|{nm}": {**ov, "data.label_noise_rate": r} for r in (0.05, 0.10)
              for nm, ov in (("NONE", E.MODE_VARIANTS["NONE"]), ("SELECTIVE", E.SEL), ("STRICT", E.STR))}
        ex += E.run_grid(cfg, ck, pipelines=("B3",), attacks=("none", "label_poisoning", "backdoor"), variants=ln, registry=registry,
                         verbose=False, allow_stream_overrides=True)
        exdf = add_pi(E.results_to_df(ex))
        exdf.to_csv(os.path.join(ck.run_dir, "results_exploratory_sweeps.csv"), index=False)
        print(f"exploratory sweeps: {len(exdf)} trials, total {time.time()-t0:.0f}s", flush=True)
    print("RUN_DIR", ck.run_dir, flush=True)
    return ck.run_dir


def run_csv_stage(root: str, csv_path: str, label_col: str, benign_label: str, time_col: str, n_classes: int,
                  entity_col: str = "", n_chunks: int = 12, chunk_size: int = 200, seeds=(0, 1, 2), bootstrap_per_class: int = 200,
                  calib_per_class: int = 100, test_per_class: int = 150, final_test_per_class: int = 150, stage_name: str = "csv_dev"):
    """OPTIONAL real-data DEVELOPMENT stage (never used for the results in this study: no real dataset was accessible).
    Chronological bootstrap -> calibration -> stream -> dev/final-test layout with duplicate removal and entity purge.
    Development mode only: a real-data CONFIRMATORY run needs its own frozen protocol and new reserved seeds.
    Returns the run directory, or raises if the file is missing (no substitute dataset is ever used)."""
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV dataset not found: {csv_path!r}. Real-data experiments are PENDING; nothing was substituted.")
    doses = json.load(open(os.path.join(root, "runs", "primary_doses.json")))
    attacks = {k: dict(v) for k, v in DEFAULT_ATTACKS.items()}
    for atk, d in doses.items():
        attacks[atk][d["param"]] = d["dose"]
    s, e = max(1, n_chunks // 4), max(2, (n_chunks * 5) // 6)
    for k in attacks:
        attacks[k].update(start=s, end=e)
    attacks["slow_poisoning"].update(start=0, end=n_chunks)
    attacks["replay_flood"].update(start=max(1, n_chunks // 2), end=max(2, n_chunks // 2) + 2)
    data = DataCfg(kind="csv", csv_path=csv_path, csv_label_col=label_col, csv_benign_label=benign_label, csv_time_col=time_col,
                   csv_entity_col=entity_col, n_classes=n_classes, n_chunks=n_chunks, chunk_size=chunk_size, schedule=(),
                   bootstrap_per_class=bootstrap_per_class, calib_per_class=calib_per_class, test_per_class=test_per_class,
                   final_test_per_class=final_test_per_class, drift_chunk=-1, drift_norm=0.0)
    cfg = RunConfig(experiment_name="evcl_validation", run_mode="development", seeds=tuple(seeds), attacks=attacks,
                    allow_simulated_sandbox=True, data=data)
    cfg = apply_overrides(cfg, {"sandbox.budget_per_chunk": max(10, chunk_size * 3 // 10)})
    for w in validate_config(cfg):
        print("WARNING:", w)
    ck = CheckpointManager(os.path.join(root, "runs", stage_name), "evcl_validation", config_hash(cfg))
    ck.write_json("config.json", cfg_to_dict(cfg))
    ck.write_json("env.json", capture_environment())
    b3 = {k: B3_VARIANTS[k] for k in ("NONE", "SELECTIVE", "STRICT", "SELECTIVE|shuffled-evidence", "STRICT|shuffled-evidence")}
    res = E.run_grid(cfg, ck, pipelines=BASE_PIPELINES, verbose=False)
    res += E.run_grid(cfg, ck, pipelines=("B3",), variants=b3, verbose=False)
    df = add_pi(E.results_to_df(res))
    assert_results_valid(df)
    df.to_csv(os.path.join(ck.run_dir, "results_main.csv"), index=False)
    print(f"CSV stage done: {len(df)} trials | {banner(df)} | NOTE telemetry is still SIMULATED unless a real adapter is supplied", flush=True)
    return ck.run_dir
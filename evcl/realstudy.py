"""evcl.realstudy -- the COMPLETE real-data study (one resumable run, seven pre-specified analyses).

Real flow features (CIC-IDS2017-style CSV) + SIMULATED sandbox telemetry. Every trial is checkpointed, so a Colab
disconnect only costs the trial that was running; re-running the same call continues where it stopped.

Blocks written to results_all.csv (column `block`):
  base        B0/B1/BR/B2/B2b baselines + B3 NONE/SELECTIVE/STRICT           (analyses 2, 4, 5, 6)
  controls    B3 shuffled-evidence, null-signal, sandbox-outage controls     (analysis 3)
  matched     BC/BR/B2b admitting the SAME fraction of data as STRICT /
              SELECTIVE did in the same (seed, attack) cell, no evidence     (analysis 2)
  sens        telemetry-fidelity LOW/HIGH and verification-budget sweeps     (analysis 7)
Provenance (analysis 1) is written to provenance.json before any trial runs.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np

from . import experiment as E
from .audit import assert_results_valid, banner
from .checkpoint import CheckpointManager, SeedRegistry
from .core import (DEFAULT_ATTACKS, DataCfg, RunConfig, apply_overrides, capture_environment, cfg_to_dict, config_hash,
                   source_hash, validate_config)
from .evaluation import add_pi

BASE_PIPES = ("B0", "B1", "BR", "B2", "B2b")
SENS_ATTACKS = ("none", "label_poisoning", "backdoor", "sandbox_evasion")
MATCH_REFS = ("STRICT", "SELECTIVE")
CONTROLS = {k: v for k, v in E.CONTROL_VARIANTS.items()}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def real_schedule(n_chunks, K, depart_class1=True):
    """Class schedule for the real-data study. The package's default schedule introduces class 5 at the LAST chunk (one chunk of exposure)
    when n_chunks=16, so no method can learn it. Here novel classes appear at n_chunks*j/(K-1) (j=1..K-3), leaving every novel class several
    chunks, and base class 1 departs at 60% of the stream so that forgetting (CFR) is measurable. Class 2 (the clean-label target) stays."""
    sch = [(0, 0, n_chunks), (1, 0, int(0.6 * n_chunks) if depart_class1 else n_chunks), (2, 0, n_chunks)]
    for j, c in enumerate(range(3, K), start=1):
        sch.append((c, max(1, int(n_chunks * j / (K - 1))), n_chunks))
    return tuple(sch)


LEARNERS = {"L0": {}, "L1": {"learner.class_balanced_updates": True},
            "L2": {"learner.class_balanced_updates": True, "learner.buffer_policy": "class_balanced"},
            # Phase 3 learner arms (other model families, same buffer policy and class-balancing as L2)
            "MLP": {"learner.algo": "mlp_dynamic", "learner.class_balanced_updates": True, "learner.buffer_policy": "class_balanced"},
            "RF": {"learner.algo": "forest_refit", "learner.class_balanced_updates": True, "learner.buffer_policy": "class_balanced"}}


def build_cfg(root, csv_path, label_col, benign_label, time_col, n_classes, entity_col, n_chunks, chunk_size, seeds,
              run_mode, bootstrap_per_class=200, calib_per_class=100, test_per_class=150, final_test_per_class=150, learner="L2"):
    doses = json.load(open(os.path.join(root, "runs", "primary_doses.json")))
    attacks = {k: dict(v) for k, v in DEFAULT_ATTACKS.items()}
    for atk, d in doses.items():
        attacks[atk][d["param"]] = d["dose"]
    s, e = max(1, n_chunks // 4), max(2, (n_chunks * 5) // 6)
    for k in attacks:
        attacks[k].update(start=s, end=e)
    attacks["slow_poisoning"].update(start=0, end=n_chunks)
    attacks["replay_flood"].update(start=max(1, n_chunks // 2), end=max(2, n_chunks // 2) + 2)
    data = DataCfg(kind="csv", csv_path=csv_path, csv_label_col=label_col, csv_benign_label=benign_label,
                   csv_time_col=time_col, csv_entity_col=entity_col, n_classes=n_classes, n_chunks=n_chunks,
                   chunk_size=chunk_size, schedule=real_schedule(n_chunks, n_classes), bootstrap_per_class=bootstrap_per_class,
                   calib_per_class=calib_per_class, test_per_class=test_per_class,
                   final_test_per_class=final_test_per_class, drift_chunk=-1, drift_norm=0.0)
    cfg = RunConfig(experiment_name="evcl_real_study", run_mode=run_mode, seeds=tuple(seeds), attacks=attacks,
                    allow_simulated_sandbox=True, data=data)
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {sorted(LEARNERS)}")
    return apply_overrides(cfg, {"sandbox.budget_per_chunk": max(10, chunk_size * 3 // 10), **LEARNERS[learner]})


HYPOTHESES = [
    "Outcome for P1-P8: seed-level mean ATTACKED headline macro-F1 (mean over the attacks run), paired over seeds, exact two-sided Wilcoxon, Holm over P1-P8.",
    "P1/P2: B3 SELECTIVE / STRICT vs B3 NONE (accept-all).  P3/P4: vs the same mode with shuffled evidence.  P5: STRICT vs STRICT with null signal.",
    "P6/P7: STRICT / SELECTIVE vs evidence-free class-balanced curation (BC) admitting the same fraction.  P8: STRICT vs trimmed-loss filter B2b.",
    "Secondary and descriptive only (no significance claim): attack success rate, legitimate-data loss (clean FRR), novel-class recall, forgetting (CFR), sensitivity to telemetry fidelity and verification budget.",
    "Inference is over independent seeds of ONE real dataset with SIMULATED sandbox telemetry; it does not generalise to other datasets or to real sandboxes.",
    "Negative or null results will be reported as such."]


def sens_variants(chunk_size):
    base = max(10, chunk_size * 3 // 10)
    v = {}
    for m, b in (("SELECTIVE", E.SEL), ("STRICT", E.STR)):
        for lvl in ("LOW", "HIGH"):
            v[f"{m}|fidelity-{lvl}"] = {**b, **E.FIDELITY[lvl]}
        for bud in (max(5, base // 3), base * 2, 10 ** 6):
            v[f"{m}|budget-{'unlimited' if bud == 10 ** 6 else bud}"] = {**b, "sandbox.budget_per_chunk": bud}
    return v


def write_provenance(root, out_dir, cfg, csv_path, prep_report_path=None, notes=""):
    """Analysis 1: which configuration produced the results. Written BEFORE any trial runs, from files on disk."""
    prov = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_mode=cfg.run_mode,
        seeds=list(map(int, cfg.seeds)), config_hash=config_hash(cfg), config=cfg_to_dict(cfg),
        code_hash=source_hash(os.path.join(root, "evcl")),
        module_sha256={os.path.basename(p): _sha(p) for p in sorted(glob.glob(os.path.join(root, "evcl", "*.py")))},
        dataset_path=csv_path, dataset_sha256=_sha(csv_path), environment=capture_environment(),
        telemetry="SIMULATED (modeled, NOT measured)", notes=notes)
    if prep_report_path and os.path.exists(prep_report_path):
        prov["prepare_report"] = json.load(open(prep_report_path))
    json.dump(prov, open(os.path.join(out_dir, "provenance.json"), "w"), indent=2, default=str)
    return prov


def run_real_study(root, csv_path, label_col="label", benign_label="BENIGN", time_col="ts", n_classes=6, entity_col="",
                   n_chunks=16, chunk_size=300, seeds=tuple(range(10)), run_mode="development",
                   stage_name="real_study", attacks=E.SCENARIOS, prep_report_path=None, notes="", verbose=True, **data_kw):
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"prepared CSV not found: {csv_path!r}")
    cfg = build_cfg(root, csv_path, label_col, benign_label, time_col, n_classes, entity_col, n_chunks, chunk_size, seeds,
                    run_mode, **data_kw)
    for w in validate_config(cfg):
        print("WARNING:", w, flush=True)
    # feasibility pre-check BEFORE anything is written or frozen: raises ValueError if a class is too small for this layout
    E.build_scenario(cfg, int(seeds[0]), "none")
    ck = CheckpointManager(os.path.join(root, "runs", stage_name), "evcl_real_study", config_hash(cfg))
    ck.write_json("config.json", cfg_to_dict(cfg))
    if run_mode == "final":      # freeze the protocol BEFORE any final trial; a later change is refused
        design = dict(blocks=["base", "controls", "sens", "matched"], attacks=list(attacks), n_chunks=n_chunks,
                      chunk_size=chunk_size, n_classes=n_classes, sens=sens_variants(chunk_size), match_refs=list(MATCH_REFS),
                      dataset_sha256=_sha(csv_path))
        pf = os.path.join(root, "runs", stage_name, "protocol_frozen.json")
        os.makedirs(os.path.dirname(pf), exist_ok=True)
        if os.path.exists(pf):
            E.check_protocol(pf, cfg, design, cfg.seeds, HYPOTHESES)
        else:
            E.freeze_protocol(pf, cfg, design, cfg.seeds, HYPOTHESES)
    write_provenance(root, ck.run_dir, cfg, csv_path, prep_report_path, notes)
    registry = SeedRegistry(os.path.join(root, "runs", stage_name), "evcl_real_study")
    t0 = time.time()
    blocks = {}

    def tag(res, name):
        for r in res:
            r["block"] = name
        blocks[name] = blocks.get(name, []) + res
        if verbose:
            print(f"[{name}] {len(res)} trials | elapsed {time.time() - t0:.0f}s", flush=True)

    # 1) baselines + B3 modes
    res = E.run_grid(cfg, ck, pipelines=BASE_PIPES, attacks=attacks, registry=registry, verbose=False)
    res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=attacks, variants=E.MODE_VARIANTS, verbose=False)
    tag(res, "base")
    # 2) evidence controls
    tag(E.run_grid(cfg, ck, pipelines=("B3",), attacks=attacks, variants=CONTROLS, verbose=False), "controls")
    # 3) sensitivity (fidelity, budget) on a subset of attacks
    sa = tuple(a for a in SENS_ATTACKS if a in attacks)
    tag(E.run_grid(cfg, ck, pipelines=("B3",), attacks=sa, variants=sens_variants(chunk_size), verbose=False), "sens")
    # 4) matched-budget curation baselines: admit the same fraction B3 admitted in the same (seed, attack) cell
    import pandas as pd
    ref = pd.DataFrame([{k: v for k, v in r.items() if k != "evals"} for r in blocks["base"]])
    ref = ref[(ref.pipeline == "B3") & ref.variant.isin(MATCH_REFS)]
    matched = []
    for seed in seeds:
        for attack in attacks:
            stream = None
            for rv in MATCH_REFS:
                row = ref[(ref.seed == seed) & (ref.attack == attack) & (ref.variant == rv)]
                if row.empty:
                    continue
                f = float(np.clip(row.accept_frac.iloc[0], 0.02, 1.0))
                plan = {"BC": {"curation_frac": f}, "BR": {"random_admit_p": f}, "B2b": {"trim_frac": float(min(0.9, 1 - f))}}
                for p, ov in plan.items():
                    vname = f"matched-{rv}"
                    key = E.trial_key(vname, p, attack, seed)
                    if ck.has_trial(key):
                        r = ck.load_trial(key)
                    else:
                        if stream is None:
                            stream = E.build_scenario(cfg, seed, attack)
                        r = E.run_trial(apply_overrides(cfg, ov), p, stream, seed, attack, vname, ck)
                    r["matched_fraction"] = f
                    # the matched parameter differs per (seed, attack) cell by design: keep the exact per-cell hash in
                    # config_hash_cell and label the row with the base-configuration hash so the audit can group it
                    r["config_hash_cell"] = r["config_hash"]
                    r["config_hash"] = config_hash(cfg)
                    matched.append(r)
    tag(matched, "matched")
    allr = [r for b in blocks.values() for r in b]
    df = add_pi(E.results_to_df(allr))
    assert_results_valid(df)
    df.to_csv(os.path.join(ck.run_dir, "results_all.csv"), index=False)
    print(f"REAL STUDY done: {len(df)} trials in {time.time() - t0:.0f}s | {banner(df)}", flush=True)
    return ck.run_dir
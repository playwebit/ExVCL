"""evcl.audit -- provenance banners, result validation and the Research Validity Audit.

The audit REPORTS what a run is and is not. It never certifies scientific correctness; the verdict is a
mechanical checklist and a human review is always required."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core import BASELINE_STATUS, FINAL_SEED_MIN, MIN_FINAL_SEEDS, PIPELINE_LABEL, RunConfig, capability_report

_TELEMETRY_TEXT = {
    "SIMULATED": "SIMULATED telemetry (modeled from ground truth + assumed noise; NOT measured sandbox data)",
    "RECORDED": "RECORDED telemetry (measured elsewhere, imported)",
    "CONTAINER": "CONTAINER (Docker/strace) telemetry",
    "NONE": "no behavioural telemetry used",
}


def banner(df) -> str:
    """One-line provenance banner placed on every table and figure."""
    b3 = df[df.pipeline == "B3"] if "pipeline" in df else df
    srcs = sorted(set(b3.telemetry_source)) if len(b3) and "telemetry_source" in b3 else ["NONE"]
    srcs_txt = " + ".join(s for s in srcs if s != "NONE") or "NONE"
    data = ",".join(sorted(set(df.data_kind))) if "data_kind" in df else "?"
    data_txt = "SYNTHETIC data" if data == "synthetic" else f"data={data}"
    n = df.seed.nunique() if "seed" in df else "?"
    mode = ",".join(sorted(set(df.run_mode))) if "run_mode" in df else "?"
    return (f"Telemetry: {srcs_txt}{' (modeled, NOT measured)' if 'SIMULATED' in srcs else ''} | {data_txt} | "
            f"B2 = dev proxy | seeds={n} ({mode}) | triage = userspace model (no eBPF/XDP) | no VM")


def assert_results_valid(df):
    """Raise if results lack provenance or are internally inconsistent."""
    need = ["telemetry_source", "baseline_status", "data_kind", "split_type", "stream_fingerprint",
            "eval_fingerprint", "run_mode", "triage_backend"]
    miss = [c for c in need if c not in df]
    if miss:
        raise AssertionError(f"results lack provenance columns: {miss}")
    ok_src = {"SIMULATED", "RECORDED", "CONTAINER", "NONE", "VM"}
    if not set(df.telemetry_source) <= ok_src:
        raise AssertionError("unknown telemetry_source tag")
    if (df[df.pipeline.isin(["B0", "B1", "B2", "B2b", "BR"])].telemetry_source != "NONE").any():
        raise AssertionError("baselines must not carry sandbox telemetry")
    b3 = df[(df.pipeline == "B3") & (df.verification_mode == "none")]
    if (b3.telemetry_source != "NONE").any():
        raise AssertionError("B3 in mode 'none' used telemetry")
    if df.PAR.dropna().between(0, 1).all() is False or not df.PAR.dropna().between(0, 1).all():
        raise AssertionError("PAR outside [0,1]")
    # identical-stream audit: all pipelines/variants share one stream per (attack, seed); evaluation data is identical
    g = df.groupby(["attack", "seed"]).stream_fingerprint.nunique()
    if g.max() != 1:
        raise AssertionError("different pipelines/variants saw different attacked streams")
    if df.groupby("seed").eval_fingerprint.nunique().max() != 1:
        raise AssertionError("evaluation data differs across attack scenarios for a seed")
    cnt = df.groupby(["variant", "pipeline", "attack"]).seed.nunique()
    if cnt.nunique() != 1:
        raise AssertionError("unequal seed counts across cells (incomplete grid)")
    if "headline_split" in df:
        fin = df[df.run_mode == "final"]
        if len(fin) and not (fin.headline_split == "final").all():
            raise AssertionError("final-mode runs must report the FINAL test period as headline")
        dev = df[df.run_mode == "development"]
        if len(dev) and (dev.headline_split != "dev").any():
            raise AssertionError("development runs must never report the final test period")
    if "stage" in df and ((df.stage == "confirmatory") != (df.run_mode == "final")).any():
        raise AssertionError("stage/run_mode mismatch")
    if "config_hash" in df and df.groupby(["stage", "variant"]).config_hash.nunique().max() > 1:
        raise AssertionError("one (stage, variant) has several configuration hashes: configurations were mixed")
    return True


def research_validity_audit(cfg: RunConfig, df, selftest_status: Optional[Dict[str, str]] = None,
                            seed_registry: Optional[Dict[str, List[int]]] = None, verbose=True,
                            protocol_frozen_before_final: Optional[bool] = None) -> Dict[str, Any]:
    cap = capability_report()
    seeds = sorted(set(int(s) for s in df.seed))
    n_seeds = len(seeds)
    tele = set(df.telemetry_source) - {"NONE"}
    synthetic, real = [], []
    if cfg.data.kind == "synthetic":
        synthetic.append("data: SYNTHETIC Gaussian-mixture generator (pipeline validation only)")
    synthetic.append("attacks: heuristic proxy implementations, not published attacks")
    if "SIMULATED" in tele:
        synthetic.append("sandbox telemetry: SIMULATED, generated from ground truth + assumed noise")
        synthetic.append("evidence-fusion calibration corpus: SIMULATED from the same generative assumptions")
    synthetic.append("labelling channel: ground truth except attacker flips"
                     + (f" + {cfg.data.label_noise_rate:.0%} natural noise" if cfg.data.label_noise_rate else
                        " (NO natural label noise: unrealistically clean)"))
    if cfg.data.kind == "csv":
        real.append(f"dataset: real flow table ({cfg.data.csv_path}); split={sorted(set(df.split_type))}")
    for s in sorted(tele - {"SIMULATED"}):
        real.append(f"sandbox telemetry source: {s} (measured)")
    real.append("userspace Python compute time / CPU of this notebook (measured; NOT data-plane, NOT sandbox latency)")
    unavailable = [f"{c['component']}: {c['status']}" for c in cap
                   if any(w in c["status"] for w in ("UNAVAILABLE", "STUB", "UNTESTED"))]
    baselines = {p: BASELINE_STATUS[p] for p in sorted(set(df.pipeline))}
    published_reproduced = False       # cannot be established by code; must be attested manually after reproduction
    st = selftest_status or {}
    tests_run = bool(st)
    tests_pass = tests_run and all(v.startswith(("PASS", "SKIP")) for v in st.values())
    tr = lambda name: st.get(name, "NOT RUN")  # noqa: E731
    sep_tests = ["test_eval_data_untouched_by_attacks", "test_periods_and_future_labels",
                 "test_duplicate_and_entity_leakage", "test_final_test_untouched_until_end"]
    tmp_tests = ["test_no_future_information", "test_triage_scores_precede_update"]
    lab_tests = ["test_pipelines_blind_to_ground_truth", "test_gate_inputs_exclude_ground_truth"]
    all_pass = lambda names: all(tr(n).startswith("PASS") for n in names)  # noqa: E731
    split_ok = all_pass(sep_tests)
    temporal_checked = all_pass(tmp_tests)
    labels_blind = all_pass(lab_tests)
    temporal_split_real = bool(df.temporal_split.all())
    residual_temporal = ["label space (K classes) is pre-allocated: the learner 'knows' how many classes will exist",
                         "chunks are atomic batches: budgeting/B2 neighbours use whole-chunk (within-chunk look-ahead)",
                         "test distribution is stationary for synthetic data (no concept drift modelled)"]
    if cfg.data.kind == "csv" and not temporal_split_real:
        residual_temporal.append("CSV split is RANDOM: near-duplicate flows can cross splits; temporal validity NOT established")
    oracle_note = ("SIMULATED verification is generated FROM the ground truth used for evaluation -> NOT independent "
                   "of evaluation labels (by construction)") if "SIMULATED" in tele else \
                  ("verification telemetry is externally produced; independence from dataset labels depends on how "
                   "payloads were matched and calibration disjointness (manual attestation required)")
    criteria = {
        "self-tests executed and all passed": tests_pass,
        f"final mode with >= {MIN_FINAL_SEEDS} reserved seeds (>= {FINAL_SEED_MIN})":
            cfg.run_mode == "final" and n_seeds >= MIN_FINAL_SEEDS and min(seeds) >= FINAL_SEED_MIN,
        "seed registry shows no dev/final seed overlap":
            bool(seed_registry) and not (set(seed_registry.get("development", [])) & set(seed_registry.get("final", []))),
        "real dataset (not synthetic)": cfg.data.kind == "csv",
        "chronological train/test split": temporal_split_real,
        "measured (non-simulated) sandbox telemetry": bool(tele) and "SIMULATED" not in tele,
        "protocol frozen before the final run": bool(protocol_frozen_before_final),
        "published baseline faithfully reproduced (manual attestation)": published_reproduced,
    }
    if not tests_pass:
        verdict = "NOT VALID: self-tests not run or failing"
    elif not criteria[f"final mode with >= {MIN_FINAL_SEEDS} reserved seeds (>= {FINAL_SEED_MIN})"]:
        verdict = "DEVELOPMENT / PIPELINE VALIDATION ONLY (not a final-mode run with >= 10 reserved seeds)"
    elif cfg.data.kind == "synthetic":
        verdict = "PIPELINE VALIDATION ONLY (synthetic data: no evidence about real NIDS)"
    elif not (criteria["chronological train/test split"] and criteria["measured (non-simulated) sandbox telemetry"]):
        verdict = ("PRELIMINARY RESEARCH ONLY: conditional on simulated telemetry and/or non-temporal split; "
                   "claims must be stated as conditional sensitivity analysis")
    elif not criteria["published baseline faithfully reproduced (manual attestation)"]:
        verdict = "PRELIMINARY RESEARCH ONLY: no faithfully reproduced published baseline"
    else:
        verdict = "MEETS the automated publication checklist; independent human review still required"
    rep = dict(
        synthetic_components=synthetic, real_measured_components=real, unavailable_components=unavailable,
        baseline_status=baselines, published_baseline_faithfully_reproduced=published_reproduced,
        n_seeds=n_seeds, seeds=seeds, run_mode=cfg.run_mode,
        train_test_separation="TESTED: no overlap/touching found (synthetic + generated-CSV fixtures)" if split_ok
        else "UNVERIFIED (required tests not run/passing)",
        temporal_leakage=("TESTED: no future-chunk information reached earlier evaluations; residual limits below"
                          if temporal_checked else "UNVERIFIED (required tests not run/passing)"),
        temporal_residual_limitations=residual_temporal,
        label_leakage=("pipelines/gate blind to ground truth (static + dynamic tests passed); " if labels_blind
                       else "UNVERIFIED pipeline blindness; ") + oracle_note,
        criteria=criteria, verdict=verdict)
    if verbose:
        bar = "=" * 92
        print(bar, "\nRESEARCH VALIDITY AUDIT   (mechanical checklist; not a certification)\n" + bar)
        print("Provenance:", banner(df))
        for k in ("synthetic_components", "real_measured_components", "unavailable_components"):
            print(f"\n{k.replace('_', ' ').upper()}:")
            for x in rep[k]:
                print("  -", x)
        print("\nBASELINE STATUS:")
        for p, s in baselines.items():
            print(f"  - {PIPELINE_LABEL[p]}: {s}")
        print("  - published defence faithfully reproduced:", "YES" if published_reproduced else "NO (none reproduced)")
        print(f"\nSEEDS: {n_seeds} ({rep['run_mode']} mode) {seeds}")
        print("TRAIN/TEST SEPARATION:", rep["train_test_separation"])
        print("TEMPORAL LEAKAGE:", rep["temporal_leakage"])
        for x in residual_temporal:
            print("   residual:", x)
        print("LABEL LEAKAGE:", rep["label_leakage"])
        print("\nCHECKLIST:")
        for k, v in criteria.items():
            print(f"  [{'x' if v else ' '}] {k}")
        print("\nVERDICT:", verdict, "\n" + bar)
    return rep
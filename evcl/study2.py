"""evcl.study2 -- EXPLORATORY follow-up studies on development seeds (never confirmatory):
  breakeven    at which attack dose does verification start to pay off? (modes x fidelity x dose)
  curation     is the unlimited-budget STRICT gain just data curation? (controls without any evidence)
  allocation   smarter use of the verification budget (random|balanced|uncertainty|novelty_balanced)
  triage_aware adaptive attacker that targets the unverified fast path (and the value of audits)
All parts: SIMULATED telemetry, synthetic stream, seeds 10-14 (fresh development seeds), no final test period."""
from __future__ import annotations

import os
import time

import pandas as pd

from . import experiment as E
from .audit import banner
from .checkpoint import CheckpointManager
from .core import apply_overrides, capture_environment, cfg_to_dict, config_hash, validate_config
from .evaluation import add_pi
from .stages import make_cfg

STUDY2_SEEDS = (10, 11, 12, 13, 14)
SEL, STR = E.SEL, E.STR
HI = E.FIDELITY["HIGH"]
UNL = {"sandbox.budget_per_chunk": 10 ** 6}
DOSES = {"label_poisoning": ("budget", [0.05, 0.10, 0.20, 0.30]),
         "replay_duplicate": ("budget", [0.05, 0.10, 0.20, 0.40, 0.80]),
         "backdoor": ("budget", [0.02, 0.05, 0.10, 0.20, 0.40])}
BREAKEVEN_VARIANTS = {"NONE": E.MODE_VARIANTS["NONE"], "SELECTIVE|MED": dict(SEL), "STRICT|MED": dict(STR),
                      "SELECTIVE|HIGH": {**SEL, **HI}, "STRICT|HIGH": {**STR, **HI}, "STRICT|MED|unlimited": {**STR, **UNL},
                      "STRICT|HIGH|balanced": {**STR, **HI, "sandbox.allocation": "balanced"},
                      "SELECTIVE|HIGH|novelty_balanced": {**SEL, **HI, "sandbox.allocation": "novelty_balanced"}}
BC_VARIANTS = {"BC frac=0.30": {"curation_frac": 0.30}, "BC frac=0.127": {"curation_frac": 0.127}}
LEARNERS = {"L0 default": {}, "L1 balanced-updates": {"learner.class_balanced_updates": True},
            "L2 balanced-updates+buffer": {"learner.class_balanced_updates": True, "learner.buffer_policy": "class_balanced"}}


def _prep(root, part, seeds, smoke):
    cfg = make_cfg(root, "development", seeds, smoke=smoke)
    cfg = apply_overrides(cfg, {"experiment_name": f"study2_{part}"})
    for w in validate_config(cfg):
        pass
    ck = CheckpointManager(os.path.join(root, "runs", "study2"), f"study2_{part}", config_hash(cfg))
    ck.write_json("config.json", cfg_to_dict(cfg))
    ck.write_json("env.json", capture_environment())
    return cfg, ck


def _finish(res, ck, part, t0):
    df = add_pi(E.results_to_df(res))          # PI only where a paired clean run with the same variant exists
    need = ["telemetry_source", "baseline_status", "data_kind", "split_type", "stream_fingerprint", "run_mode"]
    assert all(c in df for c in need), "missing provenance columns"
    assert set(df.run_mode) == {"development"} and set(df.headline_split) == {"dev"}, "study-2 parts are development-only"
    df.to_csv(os.path.join(ck.run_dir, f"results_{part}.csv"), index=False)
    print(f"[{part}] {len(df)} trials in {time.time()-t0:.0f}s | {banner(df)}", flush=True)
    return ck.run_dir


def run_part(root: str, part: str, seeds=STUDY2_SEEDS, smoke: bool = False):
    assert part in ("breakeven", "curation", "allocation", "triage_aware", "strongbaseline")
    cfg, ck = _prep(root, part, seeds, smoke)
    t0, res = time.time(), []
    attacks_all = ("label_poisoning", "clean_label", "replay_flood", "replay_duplicate", "backdoor", "slow_poisoning", "sandbox_evasion")
    if part == "breakeven":
        res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=("none",), variants=BREAKEVEN_VARIANTS, verbose=False)
        res += E.run_grid(cfg, ck, pipelines=("BC",), attacks=("none",), variants=BC_VARIANTS, verbose=False)
        for atk, (param, doses) in DOSES.items():
            if smoke:
                doses = doses[:2]
            var = {f"{v}@{atk}={d}": {**ov, f"attacks.{atk}.{param}": d} for d in doses for v, ov in BREAKEVEN_VARIANTS.items()}
            res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=(atk,), variants=var, verbose=False, allow_stream_overrides=True)
            bvar = {f"{v}@{atk}={d}": {**ov, f"attacks.{atk}.{param}": d} for d in doses for v, ov in BC_VARIANTS.items()}
            res += E.run_grid(cfg, ck, pipelines=("BC",), attacks=(atk,), variants=bvar, verbose=False, allow_stream_overrides=True)
    elif part == "curation":
        atks = ("none", "label_poisoning")
        res += E.run_grid(cfg, ck, pipelines=("B1",), attacks=atks, verbose=False)
        res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=atks, variants={"STRICT|MED|unlimited": {**STR, **UNL}, "STRICT|HIGH|unlimited": {**STR, **UNL, **HI}}, verbose=False)
        res += E.run_grid(cfg, ck, pipelines=("BR",), attacks=atks, variants={"BR p=0.127": {"random_admit_p": 0.127}, "BR p=0.30": {"random_admit_p": 0.30}}, verbose=False)
        res += E.run_grid(cfg, ck, pipelines=("BC",), attacks=atks, variants={"BC frac=0.127": {"curation_frac": 0.127}, "BC frac=0.30": {"curation_frac": 0.30}}, verbose=False)
    elif part == "allocation":
        var = {}
        for mode, mov in (("SELECTIVE", SEL), ("STRICT", STR)):
            for fid, fov in (("MED", {}), ("HIGH", HI)):
                for alloc in ("random", "balanced", "uncertainty", "novelty_balanced"):
                    var[f"{mode}|{fid}|{alloc}"] = {**mov, **fov, "sandbox.allocation": alloc}
        atks = ("none", "label_poisoning", "backdoor") if smoke else ("none",) + attacks_all
        cfg2 = apply_overrides(cfg, {"attacks.triage_aware_label": dict(budget=0.10, start=cfg.attacks["label_poisoning"]["start"], end=cfg.attacks["label_poisoning"]["end"])})
        res += E.run_grid(cfg2, ck, pipelines=("B3",), attacks=atks + (() if smoke else ("triage_aware_label",)), variants=var, verbose=False)
        res += E.run_grid(cfg2, ck, pipelines=("B1",), attacks=atks + (() if smoke else ("triage_aware_label",)), verbose=False)
    elif part == "strongbaseline":
        # HIGH-dose attacks applied globally; every method is run under three learners (default and two class-balanced baselines)
        hi = {"attacks.label_poisoning.budget": 0.30, "attacks.replay_duplicate.budget": 0.40, "attacks.backdoor.budget": 0.20}
        cfgh = apply_overrides(cfg, hi)
        atks = ("none", "label_poisoning") if smoke else ("none", "label_poisoning", "replay_duplicate", "backdoor")
        res += E.run_grid(cfgh, ck, pipelines=("B1",), attacks=atks, variants=dict(LEARNERS), verbose=False, allow_stream_overrides=True)
        res += E.run_grid(cfgh, ck, pipelines=("BC",), attacks=atks, variants={f"{k}|BC0.30": {**v, "curation_frac": 0.30} for k, v in LEARNERS.items()}, verbose=False, allow_stream_overrides=True)
        meth = {"STRICT|HIGH|balanced": {**STR, **HI, "sandbox.allocation": "balanced"},
                "SELECTIVE|HIGH|novelty_balanced": {**SEL, **HI, "sandbox.allocation": "novelty_balanced"},
                "STRICT|MED|balanced": {**STR, "sandbox.allocation": "balanced"}}
        res += E.run_grid(cfgh, ck, pipelines=("B3",), attacks=atks, variants={f"{lk}|{mk}": {**lv, **mv} for lk, lv in LEARNERS.items() for mk, mv in meth.items()}, verbose=False, allow_stream_overrides=True)
    else:  # triage_aware
        lp = cfg.attacks["label_poisoning"]
        doses = [0.05, 0.10, 0.20]
        var = {}
        for d in (doses[:1] if smoke else doses):
            for v, ov in {"NONE": E.MODE_VARIANTS["NONE"], "SELECTIVE audit=0": {**SEL, "admission.audit_rate": 0.0}, "SELECTIVE audit=5%": dict(SEL),
                          "SELECTIVE audit=20%": {**SEL, "admission.audit_rate": 0.20}, "STRICT": dict(STR)}.items():
                var[f"{v}@d{d}"] = {**ov, "attacks.triage_aware_label": dict(budget=d, start=lp["start"], end=lp["end"]),
                                    "attacks.label_poisoning.budget": d}          # the plain-label reference uses the SAME dose
        res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=("triage_aware_label", "label_poisoning"), variants=var, verbose=False, allow_stream_overrides=True)
        res += E.run_grid(cfg, ck, pipelines=("B3",), attacks=("none",), variants={"NONE": E.MODE_VARIANTS["NONE"], "SELECTIVE audit=5%": dict(SEL), "STRICT": dict(STR)}, verbose=False)
    return _finish(res, ck, part, t0)
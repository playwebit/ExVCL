"""evcl.phase3 -- Phase 3: the same gate, calibrated and evaluated PER EVIDENCE SOURCE on sessions recorded simultaneously by three collectors
(strace = replication of Phase 2 on a fresh recording; Falco = eBPF; auditd = Linux audit). Reuses the Phase 2 study runner unchanged:
each (source, stratum) is one frozen study with its own calibration on that source's pilot, its own seeds and the same decision rule."""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import phase2 as P

SOURCES = ("strace", "falco", "auditd")
SOURCE_LABEL = {"strace": "strace (ptrace; Phase 2 collector, replication)", "falco": "Falco (eBPF, modern probe)", "auditd": "Linux audit (auditd)"}


def p3_layout() -> P.P2Layout:
    """Primary layout for ~800 templated main sessions (160 per class) and a ~350-session pilot."""
    return P.P2Layout(n_chunks=8, chunk_size=60, benign_frac=0.7, novel_start=4, bootstrap_per_class=30, dev_per_class=20,
                      attack_start=2, attack_end=7)


LEARNER_ARMS = ("MLP", "RF")
ARM_VARIANTS = ("NONE", "SELECTIVE", "STRICT", "STRICT|global-random-evidence")
_ARM_OFFSET = {"L2": 0, "MLP": 10, "RF": 20}


def p3_seeds(source: str, stratum: str, learner: str = "L2"):
    """Final seeds, each used once. Per source a block of 100: primary 0-29 (L2, MLP, RF), hard 50-79 (L2, MLP, RF)."""
    i = SOURCES.index(source)
    base = 8000 + 100 * i + (50 if stratum == "hard" else 0) + _ARM_OFFSET[learner]
    return tuple(range(base, base + 10))


def stage_name(source: str, stratum: str, learner: str = "L2") -> str:
    return f"p3_{source}_{stratum}" + ("" if learner == "L2" else "_" + learner.lower())


def coverage_report(tab: pd.DataFrame) -> pd.DataFrame:
    t = tab[tab.cls != "idle"]
    return t.groupby("cls").agg(n=("covered", "size"), coverage=("covered", "mean")).round(3)


def assert_coverage(tab: pd.DataFrame, name: str, minimum: float = 0.95):
    """Pre-specified stop rule: every service-exercising class (idle and port scans excluded) must have host-evidence coverage >= 0.95."""
    t = tab[~tab.cls.isin(["idle", "port_scan"])]
    c = t.groupby("cls").covered.mean()
    bad = c[c < minimum]
    if len(bad):
        raise AssertionError(f"{name}: coverage below {minimum} for {dict(bad.round(3))}: the collector did not capture these sessions; check scope/rules before the final run")


def hypotheses_p3(source: str, stratum: str) -> list:
    base = P.HYPOTHESES_PHASE2 if stratum == "primary" else P.HYPOTHESES_PHASE2_HARD
    head = [f"PHASE 3, EVIDENCE SOURCE = {SOURCE_LABEL[source]}, STRATUM = {stratum}. Same gate, learner, nulls, statistics and decision rule as Phase 2; the original gate is calibrated ONLY on this source's own pilot recording (the M2 variants add the development recording).",
            "Sessions were recorded simultaneously by strace, Falco (eBPF) and auditd on the same VMs; the network features are identical for all sources, only the evidence differs. Each source is its own frozen study with its own seeds; no cross-source multiplicity correction is applied and the three results are reported side by side.",
            "Evidence kinds are mapped to the same seven counts for every source; syscall_total counts the events of each collector's FIXED event list and is not comparable across sources (the fusion is calibrated per source).",
            "No expectation is written about which source separates the hard-case sessions better; the strace study is a replication of the Phase 2 result on a fresh recording."]
    data = ("Data: pilot3 recording (bootstrap, calibration, development tests), dev3 recording (used ONLY for modification M2), main3 recording (stream and final test), each strictly later than the previous"
            + ("" if stratum == "primary" else "; hard3 recording is the stream of this stratum, never used for calibration")
            + ". A session whose generator did not complete (return code not 0) is excluded. Every service class of every recording must have host-evidence coverage of at least 0.95 for this source or the run stops.")
    mods = ["MODIFICATIONS (declared before the test recordings, frozen with this protocol). M1 support check: evidence that resembles neither the known-benign nor the known-malicious calibration sessions (mean distance to the 3 nearest references in standardised log-count space above the 99th percentile of the references\u2019 own distances) is reported as unavailable, so the gate defers it instead of reading it as benign. M2 calibration on the development recording: the fusion model and its thresholds are fitted on the pilot AND the development recording (which contains hard-case-type sessions, but not the type hard_mimic_ftp_bf) instead of the pilot alone. M1+M2 is the combination.",
            "Modification comparison (secondary, descriptive; does not change the verdict): for STRICT, each of M1, M2 and M1+M2 versus the original gate on the admitted-poison share and on the clean data lost, paired by seed over both attacks, exact two-sided Wilcoxon, Holm over the three modifications per source and stratum; reported next to evidence-free class-balanced curation at the SAME admitted fraction. A modification counts as an improvement only if it lowers the admitted-poison share without a clean-data loss that the evidence-free baseline at the same fraction would also have had.",
            "The development recording (dev3) is used ONLY for M2 calibration and for choosing nothing else; no tuning on pilot3, main3 or hard3."]
    return head + [data] + [h for h in base if not h.startswith("PHASE 2") and not h.startswith("Data:")] + (mods if stratum in ("primary", "hard") else [])


def hypotheses_arm(source: str, stratum: str, learner: str) -> list:
    return [f"PHASE 3 LEARNER ARM ({learner}), EVIDENCE SOURCE = {SOURCE_LABEL[source]}, STRATUM = {stratum}. Descriptive only: no verdict and no test. Same recordings, calibration, layout and variants subset (accept-all, SELECTIVE, STRICT, STRICT with global-random evidence, plus matched evidence-free curation) with the learner replaced by {'a small MLP (one hidden layer of 32 units, softmax output, replay)' if learner == 'MLP' else 'a random forest (100 trees, depth 12) refit on the replay memory after every update'}.",
            "Purpose: show how the downstream harm (attack-success rise, macro-F1) and the clean-data cost change with the model family. The admitted-poison share of STRICT does not depend on the learner (the gate decides from label and evidence), so it is not re-tested.",
            "Seeds are new and used only here."]


def freeze_all(root: str, data: Dict[str, Dict[str, pd.DataFrame]], L: P.P2Layout, sources=SOURCES, exclusion: Optional[dict] = None,
               learners=LEARNER_ARMS):
    LH = P.hard_layout(L)
    for src in sources:
        d = data[src]
        P.run_phase2_study(root, d["cal"], d["ev"], L, p3_seeds(src, "primary"), run_mode="final", stage_name=stage_name(src, "primary"),
                           exclusion_info=exclusion, hypotheses=hypotheses_p3(src, "primary"), dev=d.get("dev"), freeze_only=True)
        P.run_phase2_study(root, d["cal"], d["ev"], LH, p3_seeds(src, "hard"), run_mode="final", stage_name=stage_name(src, "hard"),
                           exclusion_info=exclusion, hard=d["hard"], hypotheses=hypotheses_p3(src, "hard"), dev=d.get("dev"), freeze_only=True)
        for ln in learners:
            for stratum, lay, hard in (("primary", L, None), ("hard", LH, d["hard"])):
                P.run_phase2_study(root, d["cal"], d["ev"], lay, p3_seeds(src, stratum, ln), run_mode="final", stage_name=stage_name(src, stratum, ln),
                                   exclusion_info=exclusion, hard=hard, hypotheses=hypotheses_arm(src, stratum, ln), learner=ln,
                                   variants_only=ARM_VARIANTS, sweep=False, extra_pipelines=(), freeze_only=True)


def run_all(root: str, data, L: P.P2Layout, sources=SOURCES, exclusion=None, run_mode="final", dev_seeds=None, verbose=True, learners=LEARNER_ARMS):
    from .phase2_analysis import analyze_phase2
    LH = P.hard_layout(L)
    verdicts = {}
    for src in sources:
        d = data[src]
        sp = p3_seeds(src, "primary") if run_mode == "final" else (dev_seeds or (0, 1))
        sh = p3_seeds(src, "hard") if run_mode == "final" else (dev_seeds or (0, 1))
        for stratum, lay, seeds, hard in (("primary", L, sp, None), ("hard", LH, sh, d["hard"])):
            nm = stage_name(src, stratum) + ("" if run_mode == "final" else "_dev")
            P.run_phase2_study(root, d["cal"], d["ev"], lay, seeds, run_mode=run_mode, stage_name=nm, exclusion_info=exclusion, hard=hard,
                               hypotheses=hypotheses_p3(src, stratum) if run_mode == "final" else None, sweep=True, extra_pipelines=("B2", "B2b"),
                               dev=d.get("dev"))
            v, _ = analyze_phase2(root, nm, title=f"Phase 3 -- {SOURCE_LABEL[src]} -- {stratum} stratum" + ("" if run_mode == "final" else " (DEVELOPMENT stand-in)"))
            verdicts[(src, stratum)] = v
            if verbose:
                print(f"[{src} / {stratum}] verdict: {v}", flush=True)
        for ln in learners:                                    # descriptive learner arms (no verdict)
            for stratum, lay, hard in (("primary", L, None), ("hard", LH, d["hard"])):
                nm = stage_name(src, stratum, ln) + ("" if run_mode == "final" else "_dev")
                seeds = p3_seeds(src, stratum, ln) if run_mode == "final" else (dev_seeds or (0, 1))
                P.run_phase2_study(root, d["cal"], d["ev"], lay, seeds, run_mode=run_mode, stage_name=nm, exclusion_info=exclusion, hard=hard,
                                   hypotheses=hypotheses_arm(src, stratum, ln) if run_mode == "final" else None, learner=ln,
                                   variants_only=ARM_VARIANTS, sweep=False, extra_pipelines=())
    return verdicts


def cross_source_report(root: str, data, L: P.P2Layout, sources=SOURCES, run_mode="final", out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or os.path.join(root, "results", "phase3_cross_source")
    os.makedirs(out_dir, exist_ok=True)
    rows, err_rows = [], []
    for src in sources:
        d = data[src]
        for stratum in ("primary", "hard"):
            nm = stage_name(src, stratum) + ("" if run_mode == "final" else "_dev")
            rd = os.path.join(root, "results", nm)
            if not os.path.exists(os.path.join(rd, "verdict.json")):
                continue
            v = json.load(open(os.path.join(rd, "verdict.json")))
            prim = pd.read_csv(os.path.join(rd, "tables", "primary_strict.csv"))
            mc = pd.read_csv(os.path.join(rd, "tables", "method_comparison.csv")).set_index("method")
            sw = pd.read_csv(os.path.join(rd, "tables", "operating_point_sweep.csv")).set_index("variant")
            full = [k for k in sw.index if k.startswith("STRICT|budget-")]
            rows.append(dict(source=src, stratum=stratum, verdict=v["verdict"], a_attacks_beaten=f'{v["a"]}/{v["n_attacks"]}', b_dose_response=f'{v["b"]}/{v["n_attacks"]}',
                             c_no_confound=v["c"], strict_share_real=prim.share_real.mean(), strict_share_nulls=f'{prim.share_null.min():.3f} to {prim.share_null.max():.3f}',
                             strict_clean_data_lost=mc.loc["STRICT real", "legit_data_loss_FRR"], accept_all_share=mc.loc["accept-all", "admitted_poison_share"],
                             strict_retention_at_full_budget=float(sw.loc[full[-1], "clean_data_retained"]) if full else np.nan))
        t = P.evidence_error_table(d["cal"], [d["ev"], d["hard"]], L)
        t.insert(0, "source", src)
        err_rows.append(t)
    summ = pd.DataFrame(rows)
    err = pd.concat(err_rows, ignore_index=True) if err_rows else pd.DataFrame()
    summ.to_csv(os.path.join(out_dir, "cross_source_summary.csv"), index=False)
    err.to_csv(os.path.join(out_dir, "evidence_error_by_source_and_type.csv"), index=False)
    piv = err.pivot_table(index=["stratum", "cls"], columns="source", values="frac_correct").round(3) if len(err) else pd.DataFrame()
    piv.to_csv(os.path.join(out_dir, "evidence_correct_by_source.csv"))
    md = ["# Phase 3 -- cross-source summary\n",
          "Each row is its own frozen study (own calibration, own seeds, same rule). Verdicts are RELATIVE to the nulls; read the absolute admitted-poison share next to accept-all.\n",
          summ.round(3).to_markdown(index=False), "\n\n## Share of sessions whose evidence is called correctly, by session type and source (gate calibrated on that source's pilot)\n",
          piv.to_markdown() if len(piv) else "no data", "\n"]
    open(os.path.join(out_dir, "REPORT_cross_source.md"), "w").write("\n".join(md))
    return os.path.join(out_dir, "REPORT_cross_source.md")


def _seed_means(df, variant, pipeline, attacks, col):
    x = df[(df.variant == variant) & (df.pipeline == pipeline) & df.attack.isin(attacks)]
    return x.groupby("seed")[col].mean().sort_index()


def modification_report(root: str, sources=SOURCES, run_mode="final", out_dir: Optional[str] = None) -> str:
    """Original gate versus M1 / M2 / M1+M2, with evidence-free class-balanced curation at the SAME admitted fraction (matched-<variant>, BC)."""
    from .evaluation import holm, paired_stats
    out_dir = out_dir or os.path.join(root, "results", "phase3_modifications")
    os.makedirs(out_dir, exist_ok=True)
    ATT = ("backdoor", "label_poisoning")
    rows, tests = [], []
    for src in sources:
        for stratum in ("primary", "hard"):
            nm = stage_name(src, stratum) + ("" if run_mode == "final" else "_dev")
            f = os.path.join(root, "runs", nm, "phase2_results_all.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            for mode in ("STRICT", "SELECTIVE"):
                var = {"original": mode, "M1": f"{mode}|support-check", "M2": f"{mode}|m2", "M1+M2": f"{mode}|m2+support-check"}
                if var["M1"] not in set(df.variant):
                    continue
                base_share = _seed_means(df, var["original"], "B3", ATT, "admitted_poison_frac")
                base_frr = _seed_means(df, var["original"], "B3", ("none",), "FRR")
                pt = []
                for lab, v in var.items():
                    share = _seed_means(df, v, "B3", ATT, "admitted_poison_frac"); frr = _seed_means(df, v, "B3", ("none",), "FRR")
                    par = _seed_means(df, v, "B3", ATT, "PAR"); f1 = _seed_means(df, v, "B3", ATT, "headline_macro_f1")
                    mshare = _seed_means(df, f"matched-{v}", "BC", ATT, "admitted_poison_frac"); mfrr = _seed_means(df, f"matched-{v}", "BC", ("none",), "FRR")
                    rows.append(dict(source=src, stratum=stratum, mode=mode, gate=lab, admitted_poison_share=share.mean(), poison_acceptance=par.mean(),
                                     clean_data_lost=frr.mean(), attacked_macro_f1=f1.mean(), evidence_free_share_same_fraction=mshare.mean(),
                                     evidence_free_clean_lost_same_fraction=mfrr.mean()))
                    if lab != "original" and mode == "STRICT":
                        a, b = base_share.reindex(share.index).values, share.values
                        c, d_ = base_frr.reindex(frr.index).values, frr.values
                        pt.append(dict(source=src, stratum=stratum, modification=lab, share_original=float(pd.Series(a).mean()), share_modified=float(pd.Series(b).mean()),
                                       **{"share_" + k: v_ for k, v_ in paired_stats(a, b).items()},
                                       clean_lost_original=float(pd.Series(c).mean()), clean_lost_modified=float(pd.Series(d_).mean()),
                                       **{"lost_" + k: v_ for k, v_ in paired_stats(d_, c).items()}))
                if pt:
                    t = pd.DataFrame(pt)
                    t["share_p_holm"] = holm(t.share_p_wilcoxon.values)
                    t["lost_p_holm"] = holm(t.lost_p_wilcoxon.values)
                    tests.append(t)
    comp = pd.DataFrame(rows)
    tt = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    comp.to_csv(os.path.join(out_dir, "modification_comparison.csv"), index=False)
    tt.to_csv(os.path.join(out_dir, "modification_tests_strict.csv"), index=False)
    md = ["# Phase 3 -- modifications versus the original gate\n",
          "Secondary and descriptive; the verdicts of the studies are not affected. share = admitted-poison share (mean over backdoor and label poisoning); clean_data_lost = share of clean data rejected or deferred (clean run). "
          "The last two columns are evidence-free class-balanced curation admitting the SAME fraction as that gate.\n",
          comp.round(3).to_markdown(index=False), "\n\n## STRICT: paired tests, modification versus original (positive share diff = the modification admits less poison; positive lost diff = it loses more clean data)\n",
          (tt[["source", "stratum", "modification", "share_original", "share_modified", "share_mean_diff", "share_ci_lo", "share_ci_hi", "share_p_holm",
               "clean_lost_original", "clean_lost_modified", "lost_mean_diff", "lost_p_holm"]].round(3).to_markdown(index=False) if len(tt) else "no data"), "\n"]
    open(os.path.join(out_dir, "REPORT_modifications.md"), "w").write("\n".join(md))
    return os.path.join(out_dir, "REPORT_modifications.md")


def learner_arm_report(root: str, sources=SOURCES, learners=("L2",) + LEARNER_ARMS, run_mode="final", out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or os.path.join(root, "results", "phase3_learner_arms")
    os.makedirs(out_dir, exist_ok=True)
    ATT = ("backdoor", "label_poisoning")
    rows = []
    for src in sources:
        for stratum in ("primary", "hard"):
            for ln in learners:
                nm = stage_name(src, stratum, ln) + ("" if run_mode == "final" else "_dev")
                f = os.path.join(root, "runs", nm, "phase2_results_all.csv")
                if not os.path.exists(f):
                    continue
                df = pd.read_csv(f)
                for v, lab in (("NONE", "accept-all"), ("SELECTIVE", "SELECTIVE"), ("STRICT", "STRICT, real evidence"), ("STRICT|global-random-evidence", "STRICT, global-random evidence"),
                               ("matched-STRICT", "class-balanced at STRICT's fraction")):
                    pipe = "BC" if v.startswith("matched") else "B3"
                    x = df[(df.variant == v) & (df.pipeline == pipe)]
                    if x.empty:
                        continue
                    c = x[x.attack == "none"]; a = x[x.attack.isin(ATT)]
                    bd = x[x.attack == "backdoor"].groupby("seed").headline_asr_all.mean() - c.groupby("seed").headline_asr_all.mean()
                    rows.append(dict(source=src, stratum=stratum, learner=ln, gate=lab, clean_macro_f1=c.headline_macro_f1.mean(), attacked_macro_f1=a.headline_macro_f1.mean(),
                                     backdoor_ASR_rise=bd.mean(), clean_data_lost=c.FRR.mean(), admitted_poison_share=a.admitted_poison_frac.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "learner_arms.csv"), index=False)
    md = ["# Phase 3 -- learner arms (descriptive; no verdict)\n", "L2 is the frozen primary softmax learner; MLP and RF are the extra model families. "
          "The admitted-poison share of STRICT does not depend on the learner; the downstream columns do.\n", t.round(3).to_markdown(index=False), "\n"]
    open(os.path.join(out_dir, "REPORT_learner_arms.md"), "w").write("\n".join(md))
    return os.path.join(out_dir, "REPORT_learner_arms.md")
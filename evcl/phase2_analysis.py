"""evcl.phase2_analysis -- report, tables, plots and the pre-specified verdict for the Phase 2 (recorded telemetry) study."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .evaluation import holm, paired_stats

ATT_ALL = ("backdoor", "label_poisoning")
NULLS = (("shuffled-evidence", "within-batch shuffled"), ("global-random-evidence", "global-random (calibration pool)"))
MIX = (0.0, 0.25, 0.5)
HARM_MIN = {"backdoor": 0.20, "label_poisoning": 0.03}


def analyze_phase2(root: str, stage: str = "phase2", out_dir: str | None = None, stamp: str = "", title: str = "Phase 2 (recorded host telemetry) -- report"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sd = os.path.join(root, "runs", stage)
    df = pd.read_csv(os.path.join(sd, "phase2_results_all.csv"))
    design = json.load(open(os.path.join(sd, "phase2_design.json")))
    out = out_dir or os.path.join(root, "results", stage)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    mode = df.run_mode.iloc[0]
    n_seeds = df.seed.nunique()
    stamp = stamp or f"RECORDED host telemetry (2 lab VMs) | poisoning simulated on recorded sessions | {mode} | {n_seeds} seeds"
    ATT = [a for a in ATT_ALL if a in df.attack.unique()]
    md = []

    def tbl(d, name, nd=3):
        d.to_csv(os.path.join(out, "tables", name + ".csv"), index=False)
        d = d.round(nd)
        return d.to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string(index=False)

    def vec(attack, variant, col, pipeline="B3"):
        x = df[(df.attack == attack) & (df.variant == variant) & (df.pipeline == pipeline)]
        return x.groupby("seed")[col].mean().sort_index()

    def clean_admit(variant):
        return vec("none", variant, "accept_frac")

    def tests(m):
        rows = []
        for a in ATT:
            real = vec(a, m, "admitted_poison_frac")
            for k, lab in NULLS:
                nul = vec(a, f"{m}|{k}", "admitted_poison_frac")
                if len(real) and len(nul):
                    rows.append(dict(mode=m, attack=a, null=lab, share_real=real.mean(), share_null=nul.mean(),
                                     clean_admit_real=clean_admit(m).mean(), clean_admit_null=clean_admit(f"{m}|{k}").mean(),
                                     **paired_stats(nul.values, real.values)))      # null - real: positive = evidence admits less poison
        t = pd.DataFrame(rows)
        if len(t):
            t["p_holm"] = holm(t.p_wilcoxon.values)
            t["evidence_helps"] = (t.p_holm < 0.05) & (t.ci_lo > 0)
        return t

    prim, sec = tests("STRICT"), tests("SELECTIVE")
    n_a = sum(1 for a in ATT if len(prim[prim.attack == a]) == len(NULLS) and bool(prim[prim.attack == a].evidence_helps.all()))
    dr_rows = []
    for m in ("STRICT", "SELECTIVE"):
        for a in ATT:
            means = [vec(a, f"{m}|global-random-evidence", "admitted_poison_frac").mean()] + \
                    [vec(a, f"{m}|mix-{q}", "admitted_poison_frac").mean() for q in MIX[1:]] + [vec(a, m, "admitted_poison_frac").mean()]
            mono = all(means[i] >= means[i + 1] - 0.005 for i in range(len(means) - 1))
            dr_rows.append(dict(mode=m, attack=a, **{"q=0 (global-random)": means[0], "q=0.25": means[1], "q=0.5": means[2], "q=1 (real)": means[3]},
                                monotone_non_increasing=mono))
    dr = pd.DataFrame(dr_rows)
    n_b = int(dr[dr["mode"] == "STRICT"].monotone_non_increasing.sum())
    conf = pd.DataFrame([dict(null=lab, clean_admit_real=clean_admit("STRICT").mean(), clean_admit_null=clean_admit(f"STRICT|{k}").mean(),
                              real_admits_at_least_as_much=bool(clean_admit("STRICT").mean() >= clean_admit(f"STRICT|{k}").mean() - 0.005))
                         for k, lab in NULLS])
    cond_c = bool(conf.real_admits_at_least_as_much.all())
    need = len(ATT)
    if n_a >= need and n_b >= need and cond_c:
        VERDICT = "CONFIRMED (recorded evidence, templated lab sessions)"
    elif n_a >= need and n_b >= need:
        VERDICT = "CONFOUNDED (real evidence admitted less clean data than a null)"
    elif n_a >= need:
        VERDICT = "PARTIAL (nulls beaten, but the dose-response is not monotone)"
    else:
        VERDICT = "NOT CONFIRMED"
    eff = []
    for a in ATT:
        col = "delta_asr_all" if a == "backdoor" else "PI_macro_f1"
        x = df[(df.variant == "NONE") & (df.pipeline == "B3") & (df.attack == a)]
        h = float(x[col].mean())
        eff.append(dict(attack=a, harm_metric=col, accept_all_harm=h, threshold=HARM_MIN[a], effective_on_accept_all=bool(h >= HARM_MIN[a])))
    eff = pd.DataFrame(eff)

    md.append(f"# {title}\n")
    md.append(f"> {stamp}\n")
    md.append(f"## Verdict (pre-specified rule): **{VERDICT}**\n")
    md.append(f"- (a) STRICT real evidence beats BOTH nulls (Holm p<0.05, CI>0) on **{n_a}/{len(ATT)}** attacks (needs all)\n"
              f"- (b) admitted-poison share non-increasing with the evidence mix q on **{n_b}/{len(ATT)}** attacks (needs all)\n"
              f"- (c) no confounding (real evidence admits at least as much clean data as each null): **{cond_c}**\n")
    md.append("**The verdict is RELATIVE: it compares real evidence with uninformative evidence. It does not say the admitted poison share is low. Read `share_real` in table 1 and the per-session evidence table for the absolute level.**\n")
    md.append("The telemetry is MEASURED on real services, but the sessions are templated lab sessions, so evidence is expected to separate very well; the poisoning is simulated "
              "on the recorded sessions. A confirmation validates the recorded-evidence channel; it is not evidence of resistance to adaptive attackers.\n")
    md.append("## 1. Primary tests -- STRICT (positive diff = the null admits MORE poison = evidence helps)\n\n" + tbl(prim, "primary_strict") + "\n")
    md.append("## 2. Confounding check -- clean data admitted (clean run)\n\n" + tbl(conf, "confounding_check") + "\n")
    md.append("## 3. Dose-response of admitted-poison share (evidence mix q; mean over seeds)\n\n" + tbl(dr, "dose_response_share") + "\n")
    md.append("## 4. Attack effectiveness on accept-all (descriptive; does not change the verdict)\n\n" + tbl(eff, "attack_effectiveness") + "\n")
    md.append("## 5. Secondary, descriptive -- SELECTIVE (same tests, no verdict)\n\n" + (tbl(sec, "secondary_selective") if len(sec) else "no data") + "\n")

    meths = [("NONE", "B3", "accept-all"), ("SELECTIVE", "B3", "SELECTIVE real"), ("SELECTIVE|shuffled-evidence", "B3", "SELECTIVE shuffled"),
             ("SELECTIVE|global-random-evidence", "B3", "SELECTIVE global-random"), ("SELECTIVE|sandbox-outage", "B3", "SELECTIVE outage"),
             ("STRICT", "B3", "STRICT real"), ("STRICT|shuffled-evidence", "B3", "STRICT shuffled"),
             ("STRICT|global-random-evidence", "B3", "STRICT global-random"), ("STRICT|sandbox-outage", "B3", "STRICT outage"),
             ("base", "B2", "B2 kNN label sanitiser"), ("base", "B2b", "B2b loss trimming"),
             ("matched-SELECTIVE", "BR", "random throttle @SELECTIVE"), ("matched-SELECTIVE", "BC", "class-balanced @SELECTIVE"),
             ("matched-STRICT", "BR", "random throttle @STRICT"), ("matched-STRICT", "BC", "class-balanced @STRICT")]
    rows = []
    for v, p, lab in meths:
        x = df[(df.variant == v) & (df.pipeline == p)]
        if x.empty:
            continue
        c = x[x.attack == "none"]
        a = x[x.attack.isin(ATT)]
        bd = x[x.attack == "backdoor"].groupby("seed").headline_asr_all.mean() - c.groupby("seed").headline_asr_all.mean() if "backdoor" in ATT else pd.Series(dtype=float)
        rows.append(dict(method=lab, legit_data_loss_FRR=c.FRR.mean(), clean_macroF1=c.headline_macro_f1.mean(), PAR_poison_acceptance=a.PAR.mean(),
                         admitted_poison_share=a.admitted_poison_frac.mean(), backdoor_ASR_rise=bd.mean() if len(bd) else float("nan")))
    mc = pd.DataFrame(rows)
    md.append("## 6. Gate-level comparison of all methods (mean over the attacks)\n\nPAR = share of poisoned observations admitted; FRR = share of clean observations rejected or deferred. Down-left is better. "
              "STRICT's outage row admits nothing by design (fail-safe), so its poison share is undefined.\n\n" + tbl(mc, "method_comparison") + "\n")

    def att_f1(variant):
        x = df[(df.variant == variant) & (df.pipeline == "B3") & df.attack.isin(ATT)]
        return x.groupby("seed").headline_macro_f1.mean().sort_index()
    drows = []
    base_f1 = att_f1("NONE")
    for m in ("SELECTIVE", "STRICT"):
        a_ = att_f1(m)
        if len(a_) and len(base_f1):
            drows.append(dict(contrast=f"{m} minus accept-all", f1_method=a_.mean(), f1_accept_all=base_f1.mean(), **paired_stats(a_.values, base_f1.values)))
    dd = pd.DataFrame(drows)
    if len(dd):
        dd["p_holm"] = holm(dd.p_wilcoxon.values)
    md.append("## 7. Downstream (descriptive): attacked macro-F1 versus accept-all\n\n" + (tbl(dd, "downstream") if len(dd) else "no data") + "\n")

    sw_names = [v for v in df.variant.unique() if "|budget-" in v]
    if sw_names:
        rows = []
        for v in ["STRICT"] + sorted(x for x in sw_names if x.startswith("STRICT")) + ["SELECTIVE"] + sorted(x for x in sw_names if x.startswith("SELECTIVE")) + ["NONE"]:
            x = df[(df.variant == v) & (df.pipeline == "B3")]
            if x.empty:
                continue
            c = x[x.attack == "none"]
            a = x[x.attack.isin(ATT)]
            rows.append(dict(variant=v, clean_data_retained=1 - c.FRR.mean(), clean_macroF1=c.headline_macro_f1.mean(),
                             admitted_poison_share=a.admitted_poison_frac.mean(), PAR_poison_acceptance=a.PAR.mean()))
        sw = pd.DataFrame(rows)
        md.append("## 8. Operating-point sweep (descriptive, pre-specified)\n\n" + tbl(sw, "operating_point_sweep") + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for _, r in mc.iterrows():
        ax.scatter(r.legit_data_loss_FRR, r.PAR_poison_acceptance, s=45)
        ax.annotate(r.method, (r.legit_data_loss_FRR, r.PAR_poison_acceptance), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("legitimate-data loss (clean-run FRR)"); ax.set_ylabel("poison acceptance rate (lower = safer)")
    ax.set_title("Phase 2: poison let in vs clean data lost (down-left is better)", fontsize=9)
    fig.text(0.01, 0.005, stamp, fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(out, "plots", "gate_tradeoff.png"), dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, max(1, len(ATT)), figsize=(4.4 * max(1, len(ATT)), 3.6), squeeze=False)
    for j, a in enumerate(ATT):
        ax = axes[0][j]
        for m, st in (("STRICT", "o-"), ("SELECTIVE", "s--")):
            ys = [vec(a, f"{m}|global-random-evidence", "admitted_poison_frac").mean()] + [vec(a, f"{m}|mix-{q}", "admitted_poison_frac").mean() for q in MIX[1:]] + [vec(a, m, "admitted_poison_frac").mean()]
            ax.plot([0, 0.25, 0.5, 1.0], ys, st, label=m)
        ax.set_title(a, fontsize=8); ax.set_xlabel("share q of observations with REAL evidence", fontsize=7); ax.set_ylabel("poison share of admitted data", fontsize=7)
        ax.legend(fontsize=6)
    fig.text(0.01, 0.003, stamp, fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(out, "plots", "dose_response_share.png"), dpi=140); plt.close(fig)
    md.append("## How to read this\n\n- CONFIRMED needs all three conditions; PARTIAL, CONFOUNDED and NOT CONFIRMED are reported exactly as they are.\n"
              "- Only host-visible attacks (SSH and FTP brute force) are in the gate study; port scans are reported separately as a host-evidence limitation.\n"
              "- The sessions are templated, so near-perfect separability is expected; the study does not test ambiguous evidence.\n")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(md))
    json.dump(dict(verdict=VERDICT, a=n_a, b=n_b, c=cond_c, n_attacks=len(ATT)), open(os.path.join(out, "verdict.json"), "w"))
    return VERDICT, out
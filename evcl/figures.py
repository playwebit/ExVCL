"""evcl.figures -- publication-style figures (PNG 300 dpi + vector PDF, no footers; captions belong in the paper) for Phase 2 and Phase 3.
All figures are computed from the result files of the studies (runs/<stage>/phase2_results_all.csv, results/<stage>/tables/*.csv)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"accept": "#B03A2E", "real": "#1F5F99", "shuffled": "#E08E0B", "global": "#7F8C8D", "m1": "#2E8B57", "m2": "#7B4FA3", "m12": "#C2185B",
     "free": "#34495E", "L2": "#1F5F99", "MLP": "#E08E0B", "RF": "#2E8B57", "bd": "#1F5F99", "lp": "#C2185B"}
ATT = ("backdoor", "label_poisoning")
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return name


def _load(root, stage):
    f = os.path.join(root, "runs", stage, "phase2_results_all.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def _seed_vec(df, variant, pipeline="B3", attacks=ATT, col="admitted_poison_frac"):
    x = df[(df.variant == variant) & (df.pipeline == pipeline) & df.attack.isin(attacks)]
    return x.groupby("seed")[col].mean() if len(x) else pd.Series(dtype=float)


def _ms(v):
    v = v.dropna()
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0) if len(v) else (np.nan, 0.0)


def _grid(n, ncols=3):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def fig_share_by_condition(panels: List[Tuple[str, pd.DataFrame]], out_dir, name="fig_admitted_poison_share"):
    """STRICT's admitted-poison share with real evidence versus the two nulls and accept-all (mean over attacks; error bar = SE over seeds)."""
    nr, nc = _grid(len(panels))
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.6 * nr), squeeze=False, sharey=True)
    conds = [("NONE", "Accept-all", C["accept"]), ("STRICT", "STRICT\nreal", C["real"]), ("STRICT|shuffled-evidence", "STRICT\nshuffled", C["shuffled"]),
             ("STRICT|global-random-evidence", "STRICT\nglobal-\nrandom", C["global"])]
    for ax, (title, df) in zip(axes.ravel(), panels):
        ms = [_ms(_seed_vec(df, v)) for v, _, _ in conds]
        ax.bar(range(len(conds)), [m * 100 for m, _ in ms], yerr=[e * 100 for _, e in ms], color=[c for *_, c in conds], capsize=2.5, width=0.7, edgecolor="none")
        for i, (m, e) in enumerate(ms):
            if np.isfinite(m):
                ax.text(i, (m + e) * 100 + 0.5, f"{m * 100:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([l for _, l, _ in conds]); ax.set_title(title)
    for ax in axes[:, 0]:
        ax.set_ylabel("poison share of admitted data (%)")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_dose_response(panels, out_dir, name="fig_dose_response"):
    nr, nc = _grid(len(panels))
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.6 * nr), squeeze=False, sharey=True)
    q = [0, 0.25, 0.5, 1.0]
    for ax, (title, df) in zip(axes.ravel(), panels):
        for a, lab, col in (("backdoor", "backdoor", C["bd"]), ("label_poisoning", "label poisoning", C["lp"])):
            vs = [_seed_vec(df, "STRICT|global-random-evidence", attacks=(a,)), _seed_vec(df, "STRICT|mix-0.25", attacks=(a,)),
                  _seed_vec(df, "STRICT|mix-0.5", attacks=(a,)), _seed_vec(df, "STRICT", attacks=(a,))]
            ms = [_ms(v) for v in vs]
            ax.errorbar(q, [m * 100 for m, _ in ms], yerr=[e * 100 for _, e in ms], marker="o", ms=3.5, lw=1.3, capsize=2, color=col, label=lab)
        ax.set_xticks(q); ax.set_xlabel("share q with real evidence"); ax.set_title(title)
    for ax in axes[:, 0]:
        ax.set_ylabel("poison share of admitted data (%)")
    axes[0][0].legend(frameon=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, name)


_TRADE = [("accept-all", C["accept"], "s", True, "accept-all"), ("SELECTIVE real", C["real"], "o", True, "SELECTIVE, real evidence"),
          ("SELECTIVE shuffled", C["shuffled"], "o", True, "SELECTIVE, shuffled"), ("SELECTIVE global-random", C["global"], "o", True, "SELECTIVE, global-random"),
          ("STRICT real", C["real"], "D", True, "STRICT, real evidence"), ("STRICT shuffled", C["shuffled"], "D", True, "STRICT, shuffled"),
          ("STRICT global-random", C["global"], "D", True, "STRICT, global-random"),
          ("random throttle @SELECTIVE", C["free"], "v", False, "random throttling (same fraction)"), ("random throttle @STRICT", C["free"], "v", False, None),
          ("class-balanced @SELECTIVE", C["free"], "^", False, "class-balanced curation (same fraction)"), ("class-balanced @STRICT", C["free"], "^", False, None),
          ("B2 kNN label sanitiser", "#8E44AD", "x", True, "sanitiser B2 (kNN)"), ("B2b loss trimming", "#8E44AD", "+", True, "sanitiser B2b (loss trimming)")]


def fig_tradeoff(panels_tables: List[Tuple[str, pd.DataFrame]], out_dir, name="fig_gate_tradeoff"):
    """Poison acceptance versus clean data lost, one point per method (method_comparison table). Down-left is better."""
    nr, nc = _grid(len(panels_tables))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.9 * nr + 0.9), squeeze=False)
    handles = {}
    for ax, (title, t) in zip(axes.ravel(), panels_tables):
        tt = t.set_index("method")
        for key, col, mk, filled, lab in _TRADE:
            if key in tt.index:
                r = tt.loc[key]
                kw = dict(color=col, marker=mk, s=34, zorder=3, linewidths=1.3)
                if mk in ("x", "+"):
                    h = ax.scatter(r.legit_data_loss_FRR * 100, r.PAR_poison_acceptance * 100, **kw)
                else:
                    h = ax.scatter(r.legit_data_loss_FRR * 100, r.PAR_poison_acceptance * 100, facecolors=col if filled else "none", edgecolors=col, marker=mk, s=34, zorder=3, linewidths=1.3)
                if lab:
                    handles[lab] = h
        ax.set_xlabel("clean data lost (%)"); ax.set_ylabel("poison acceptance (%)"); ax.set_title(title); ax.set_xlim(-4, 104); ax.set_ylim(-4, 104)
    for ax in axes.ravel()[len(panels_tables):]:
        ax.axis("off")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    return _save(fig, out_dir, name)


def fig_evidence_correct(err: pd.DataFrame, out_dir, name="fig_evidence_by_session_type"):
    """Share of sessions of each type whose evidence is called correctly by a gate calibrated on the pilot. One source: horizontal bars; several: heat map."""
    if "source" not in err.columns:
        err = err.assign(source="evidence")
    piv = err.pivot_table(index=["stratum", "cls"], columns="source", values="frac_correct")
    order = [("templated", c) for c in ("benign_web", "benign_ssh", "benign_ftp", "ssh_bruteforce", "ftp_bruteforce", "port_scan", "idle")] + \
            [("hard", c) for c in ("hard_benign_admin", "hard_benign_typos", "hard_slow_ssh_bf", "hard_slow_ftp_bf", "hard_mimic_ssh_bf", "hard_mimic_ftp_bf")]
    piv = piv.reindex([i for i in order if i in piv.index])
    labels = {"benign_web": "benign web", "benign_ssh": "benign SSH", "benign_ftp": "benign FTP", "ssh_bruteforce": "SSH brute force", "ftp_bruteforce": "FTP brute force",
              "port_scan": "port scan", "idle": "idle", "hard_benign_admin": "benign admin login", "hard_benign_typos": "benign login, mistyped passwords",
              "hard_slow_ssh_bf": "slow SSH brute force", "hard_slow_ftp_bf": "slow FTP brute force", "hard_mimic_ssh_bf": "mimicry SSH brute force", "hard_mimic_ftp_bf": "mimicry FTP brute force"}
    ylab = [("hard: " if s_ == "hard" else "") + labels.get(c, c) for s_, c in piv.index]
    nt = sum(1 for s_, _ in piv.index if s_ == "templated")
    if piv.shape[1] == 1:
        v = piv.iloc[:, 0].values * 100
        fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(piv) + 0.9))
        cols = [C["real"] if x >= 90 else C["shuffled"] if x >= 40 else C["accept"] for x in v]
        ax.barh(range(len(piv)), v, color=cols, height=0.7)
        for i, x in enumerate(v):
            ax.text(x + 1.5, i, f"{x:.0f}", va="center", fontsize=7)
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(ylab); ax.invert_yaxis(); ax.set_xlim(0, 112); ax.set_xlabel("evidence called correctly (%)")
        ax.axhline(nt - 0.5, color="black", lw=0.8, ls="--")
    else:
        fig, ax = plt.subplots(figsize=(1.6 + 1.2 * piv.shape[1] + 2.0, 0.34 * len(piv) + 1.1))
        im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv))); ax.set_yticklabels(ylab)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if np.isfinite(piv.values[i, j]):
                    ax.text(j, i, f"{piv.values[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
        ax.axhline(nt - 0.5, color="black", lw=1.0); ax.spines[["left", "bottom"]].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("evidence called correctly (%)", fontsize=7.5); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "50", "100"])
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_modifications(comp: pd.DataFrame, out_dir, name="fig_modifications"):
    """STRICT: admitted-poison share versus clean data lost for the original gate and M1, M2, M1+M2; hollow marker = evidence-free class-balanced curation at the same admitted fraction."""
    c = comp[comp["mode"] == "STRICT"]
    keys = [(s, t) for s in c.source.unique() for t in ("primary", "hard") if len(c[(c.source == s) & (c.stratum == t)])]
    nr, nc = _grid(len(keys))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.8 * nr), squeeze=False)
    col = {"original": C["real"], "M1": C["m1"], "M2": C["m2"], "M1+M2": C["m12"]}
    for ax, (s, t) in zip(axes.ravel(), keys):
        d = c[(c.source == s) & (c.stratum == t)]
        for _, r in d.iterrows():
            ax.scatter(r.clean_data_lost * 100, r.admitted_poison_share * 100, s=40, color=col[r.gate], zorder=3, label=r.gate)
            if np.isfinite(r.evidence_free_share_same_fraction):
                ax.scatter(r.evidence_free_clean_lost_same_fraction * 100, r.evidence_free_share_same_fraction * 100, s=40, facecolors="none", edgecolors=col[r.gate], zorder=3)
                ax.plot([r.clean_data_lost * 100, r.evidence_free_clean_lost_same_fraction * 100], [r.admitted_poison_share * 100, r.evidence_free_share_same_fraction * 100],
                        color=col[r.gate], lw=0.7, alpha=0.6)
        ax.set_title(f"{s} / {t}"); ax.set_xlabel("clean data lost (%)"); ax.set_ylabel("poison share of admitted (%)")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.legend(list(dict(zip(l, h)).values()), list(dict(zip(l, h)).keys()), loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               title="filled marker: the gate;  hollow marker: evidence-free class-balanced curation at the same admitted fraction", title_fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, out_dir, name)


def fig_learner_arms(arms: pd.DataFrame, out_dir, name="fig_learner_arms"):
    """Backdoor attack-success rise by gate and learner family (softmax L2, MLP, random forest)."""
    keys = [(s, t) for s in arms.source.unique() for t in ("primary", "hard") if len(arms[(arms.source == s) & (arms.stratum == t)])]
    nr, nc = _grid(len(keys))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr), squeeze=False, sharey=True)
    gates = ["accept-all", "SELECTIVE", "STRICT, real evidence", "STRICT, global-random evidence", "class-balanced at STRICT's fraction"]
    short = ["accept-all", "SELECTIVE", "STRICT\nreal", "STRICT\nglobal-rand.", "class-bal.\n@STRICT"]
    learners = [l for l in ("L2", "MLP", "RF") if l in set(arms.learner)]
    w = 0.8 / max(1, len(learners))
    for ax, (s, t) in zip(axes.ravel(), keys):
        d = arms[(arms.source == s) & (arms.stratum == t)]
        for k, ln in enumerate(learners):
            vals = [float(d[(d.learner == ln) & (d.gate == g)].backdoor_ASR_rise.mean()) if len(d[(d.learner == ln) & (d.gate == g)]) else np.nan for g in gates]
            ax.bar(np.arange(len(gates)) + (k - (len(learners) - 1) / 2) * w, vals, w, color=C[ln], label={"L2": "softmax", "MLP": "MLP", "RF": "random forest"}[ln])
        ax.set_xticks(range(len(gates))); ax.set_xticklabels(short, fontsize=6.5); ax.set_title(f"{s} / {t}"); ax.axhline(0, color="black", lw=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("backdoor attack-success rise")
    axes.ravel()[0].legend(frameon=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def make_phase2_figures(root: str, out_dir: Optional[str] = None, stages=(("Primary study (templated sessions)", "phase2_final"), ("Hard-case stratum", "phase2_hard_final")), err_csv: Optional[str] = None):
    out_dir = out_dir or os.path.join(root, "figures", "phase2")
    panels = [(t, _load(root, s)) for t, s in stages if _load(root, s) is not None]
    tabs = [(t, pd.read_csv(os.path.join(root, "results", s, "tables", "method_comparison.csv"))) for t, s in stages if os.path.exists(os.path.join(root, "results", s, "tables", "method_comparison.csv"))]
    made = [fig_share_by_condition(panels, out_dir), fig_dose_response(panels, out_dir), fig_tradeoff(tabs, out_dir)]
    if err_csv and os.path.exists(err_csv):
        made.append(fig_evidence_correct(pd.read_csv(err_csv), out_dir, "fig_evidence_by_session_type"))
    return out_dir, made


def make_phase3_figures(root: str, sources: Sequence[str], out_dir: Optional[str] = None, run_mode="final"):
    from .phase3 import stage_name, LEARNER_ARMS
    out_dir = out_dir or os.path.join(root, "figures", "phase3")
    suf = "" if run_mode == "final" else "_dev"
    panels, tabs = [], []
    for s in sources:
        for t in ("primary", "hard"):
            nm = stage_name(s, t) + suf
            df = _load(root, nm)
            if df is not None:
                panels.append((f"{s} / {t}", df))
                f = os.path.join(root, "results", nm, "tables", "method_comparison.csv")
                if os.path.exists(f):
                    tabs.append((f"{s} / {t}", pd.read_csv(f)))
    made = [fig_share_by_condition(panels, out_dir), fig_dose_response(panels, out_dir), fig_tradeoff(tabs, out_dir)]
    f = os.path.join(root, "results", "phase3_cross_source", "evidence_error_by_source_and_type.csv")
    if os.path.exists(f):
        made.append(fig_evidence_correct(pd.read_csv(f), out_dir))
    f = os.path.join(root, "results", "phase3_modifications", "modification_comparison.csv")
    if os.path.exists(f):
        made.append(fig_modifications(pd.read_csv(f), out_dir))
    f = os.path.join(root, "results", "phase3_learner_arms", "learner_arms.csv")
    if os.path.exists(f):
        made.append(fig_learner_arms(pd.read_csv(f), out_dir))
    return out_dir, made
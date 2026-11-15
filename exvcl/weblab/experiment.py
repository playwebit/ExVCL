"""exvcl.weblab.experiment -- the web-lab poisoning experiment with a budget sweep and the attribution controls.

Pipeline per event:  request -> independent upstream text model (observed label) -> [poisoning of the labelling channel]
                     -> ExVCL gate (evidence = the request executed in a strace sandbox) -> continual learner (softmax on request features)

Conditions: clean | labelflip (random share of observed labels flipped) | backdoor (attack requests carry the marker `ref=zq7` and are labelled benign).
Modes: none (accept-all), selective, strict, and strict with evidence controls (shuffled / global_random) at every verification budget.
Metrics are per seed; the report gives means, paired tests (Holm-corrected) and the evidence quality per request family.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from evcl.admission import REASON_NAMES
from ..api import ExVCL
from ..attribution import ChunkVerifier, admitted_poison_share, clean_retention
from ..evidence import EvidenceModel
from . import payloads as P
from . import sandbox, target
from .features import learner_features, matrix
from .upstream import UpstreamModel

R_MISMATCH = REASON_NAMES.index("verdict_label_mismatch")
R_DEFER = REASON_NAMES.index("budget_deferred")
R_FAST = REASON_NAMES.index("fast_admit")


def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def _paired(a, b, n_boot=2000, seed=0):
    """b - a over seeds: mean, bootstrap 95% CI, Wilcoxon p (nan if all differences are zero)."""
    from scipy.stats import wilcoxon
    d = np.asarray(b, float) - np.asarray(a, float); rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    p = float("nan") if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return dict(mean=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), p=p, n=len(d))


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns); L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(L)


def run_experiment(out_dir: str, n_events: int = 400, seeds: int = 20, budgets=(0.3, 0.5, 0.7, 1.0), poison_rate: float = 0.2, backdoor_rate: float = 0.05, chunk: int = 20,
                   workers: int = 2, seed: int = 0, learner: str = "softmax", upstream_attack_families=("A1", "A2", "A3"),
                   stream_attack_families=None, stream_benign_families=None, progress=print) -> Dict:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    target.ensure_lab()
    # ---- data sets (distinct seeds, never shared)
    upstream = UpstreamModel(seed).fit(P.generate(600, seed + 11, attack_families=upstream_attack_families))
    up_eval = upstream.evaluate(P.generate(1000, seed + 12))
    boot = P.generate(60, seed + 21)
    cal = P.generate(80, seed + 22, attack_families=P.EFFECTIVE_EVIDENCE)
    stream = P.generate(n_events, seed + 23, attack_families=stream_attack_families, benign_families=stream_benign_families)
    test_clean = P.generate(300, seed + 24)
    test_trig = P.generate(150, seed + 25, attack_share=1.0)
    progress(f"upstream model: accuracy {up_eval['accuracy']:.3f}, attack recall {up_eval['attack_recall']:.3f}, benign false positives {up_eval['benign_false_positive']:.3f}")
    # ---- evidence by sandbox execution (cached)
    names = [e.name for e in cal] + [e.name for e in stream]
    cache = os.path.join(out_dir, "evidence_cache.npz")
    F = ok = sec = None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == names:
            F, ok, sec = z["F"], z["ok"], z["sec"]; progress("evidence: reused cache")
    if F is None:
        progress(f"evidence: running {len(names)} requests in the strace sandbox ({workers} workers) ...")
        F, ok, sec = sandbox.collect_evidence(names, workers, progress=lambda i, n: progress(f"  sandbox {i}/{n}"))
        np.savez(cache, names=np.array(names, dtype=object), F=F, ok=ok, sec=sec)
    F_cal, ok_cal, F_st, ok_st = F[:len(cal)], ok[:len(cal)], F[len(cal):], ok[len(cal):]
    sandbox_s = float(sec.mean())
    y_cal = np.array([e.ground_truth for e in cal])
    ev = EvidenceModel(seed=seed, drop=("syscall_total",)).fit(F_cal[ok_cal], y_cal[ok_cal])
    # ---- evidence quality per request family (what the verdict says about each kind of request)
    verd = ev.verdicts(F_st, ok_st).verdict
    fam_rows = []
    for fam in sorted({e.family for e in stream}):
        m = np.array([e.family == fam for e in stream])
        fam_rows.append(dict(family=fam, truth="attack" if fam.startswith("A") else "benign", n=int(m.sum()),
                             read_malicious=float((verd[m] == 1).mean()), read_benign=float((verd[m] == 0).mean()),
                             undecided=float(np.isin(verd[m], (2, 3)).mean())))
    fam_df = pd.DataFrame(fam_rows)
    # ---- features
    X_boot, y_boot = matrix(boot), np.array([e.ground_truth for e in boot])
    X_st0 = matrix(stream); y_st = np.array([e.ground_truth for e in stream]); up_label = upstream.predict(stream)
    X_te, y_te = matrix(test_clean), np.array([e.ground_truth for e in test_clean])
    X_trig = np.vstack([learner_features(e.name, P.MARKER) for e in test_trig])
    marker_col = int(np.flatnonzero(np.array([n == "marker" for n in __import__("exvcl.weblab.features", fromlist=["x"]).FEATURE_NAMES]))[0])
    n = len(stream)
    cond_variants = []                                                      # (mode, budget, control)
    for b in budgets:
        for mode in ("selective", "strict"):
            cond_variants.append((mode, b, "real"))
    ctrl_variants = [("strict", b, c) for b in budgets for c in ("shuffled", "global_random")]
    rows: List[Dict] = []
    for s in range(seeds):
        for pcond in ("clean", "labelflip", "backdoor"):
            rng = np.random.default_rng(100_000 * (seed + 1) + 1000 * s + {"clean": 0, "labelflip": 1, "backdoor": 2}[pcond])
            order = rng.permutation(n)
            X = X_st0[order].copy(); yg = y_st[order]; obs = up_label[order].copy(); Fo = F_st[order]; oko = ok_st[order]
            poisoned = np.zeros(n, bool)
            if pcond == "labelflip":
                idx = rng.choice(n, int(round(poison_rate * n)), replace=False); obs[idx] = 1 - obs[idx]; poisoned[idx] = obs[idx] != yg[idx]   # a flip that corrects an upstream error is not poison
            elif pcond == "backdoor":
                cand = np.flatnonzero(yg == 1); idx = rng.choice(cand, min(int(round(backdoor_rate * n)), len(cand)), replace=False)
                obs[idx] = 0; poisoned[idx] = True
                for j in idx:
                    X[j, marker_col] = 1.0
            wrong = obs != yg
            variants = [("none", None, "real")] + cond_variants + ([] if pcond == "clean" else ctrl_variants)
            for mode, b, control in variants:
                ex = ExVCL(X.shape[1], mode, ev if mode != "none" else None, learner, budget=(b if b is not None else 0.3), audit_rate=0.05, seed=s)
                ex.bootstrap(X_boot, y_boot)
                if mode == "selective":
                    ex.calibrate_triage(X_boot[y_boot == 0])
                vr = np.random.default_rng(777 + s)
                admit = np.zeros(n, bool); route = np.full(n, -1); reason = np.zeros(n, int)
                for a in range(0, n, chunk):
                    sl = np.arange(a, min(a + chunk, n))
                    ver = None if mode == "none" else ChunkVerifier(Fo[sl], oko[sl], control=control, pool=F_cal, rng=vr)
                    r = ex.process_chunk(X[sl], obs[sl], ver)
                    admit[sl] = r.admit; route[sl] = r.route; reason[sl] = r.reason
                pred = ex.predict(X_te)
                att = y_te == 1
                row = dict(seed=s, poison=pcond, mode=mode, budget=b, control=control, n=n, n_poison=int(poisoned.sum()),
                           poison_admitted=int((admit & poisoned).sum()), poison_rejected=int((poisoned & (reason == R_MISMATCH)).sum()),
                           poison_deferred=int((poisoned & (reason == R_DEFER)).sum()), poison_fast=int((poisoned & (reason == R_FAST)).sum()),
                           poison_admitted_share=admitted_poison_share(admit, poisoned), clean_retention=clean_retention(admit, poisoned),
                           mislabelled=int(wrong.sum()), mislabelled_admitted_share=float(admit[wrong].mean()) if wrong.any() else float("nan"),
                           correct_label_retention=float(admit[~wrong].mean()), verified=int((route == 1).sum()),
                           macro_f1=float(f1_score(y_te, pred, average="macro")), attack_recall=float((pred[att] == 1).mean()),
                           asr=float((ex.predict(X_trig) == 0).mean()), est_sandbox_s=float((route == 1).sum() * sandbox_s))
                rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    # ---- summaries
    runs["budget"] = runs["budget"].fillna(-1.0)
    keys = ["poison", "mode", "control", "budget"]
    summ = runs.groupby(keys).agg(poison_admitted=("poison_admitted", "sum"), n_poison=("n_poison", "sum"), poison_rejected=("poison_rejected", "sum"),
                                  poison_deferred=("poison_deferred", "sum"), poison_fast=("poison_fast", "sum"),
                                  share=("poison_admitted_share", "mean"), mislab_adm=("mislabelled_admitted_share", "mean"),
                                  retention=("clean_retention", "mean"), correct_ret=("correct_label_retention", "mean"),
                                  f1=("macro_f1", "mean"), asr=("asr", "mean"), verified=("verified", "mean")).reset_index()
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    # ---- paired tests
    tests = []
    def sel(pc, mode, b, ctrl="real"):
        d = runs[(runs.poison == pc) & (runs["mode"] == mode) & (runs.control == ctrl) & (np.isclose(runs.budget, b if b is not None else -1.0))]
        return d.sort_values("seed")
    for pc in ("labelflip", "backdoor"):
        for b in budgets:
            st, no = sel(pc, "strict", b), sel(pc, "none", None)
            for metric in ("asr", "macro_f1", "poison_admitted_share"):
                tests.append(dict(poison=pc, budget=b, comparison=f"strict minus none: {metric}", **_paired(no[metric].values, st[metric].values)))
            tests.append(dict(poison=pc, budget=b, comparison="global_random minus real (strict): poison_admitted_share",
                              **_paired(st.poison_admitted_share.values, sel(pc, "strict", b, "global_random").poison_admitted_share.values)))
            tests.append(dict(poison=pc, budget=b, comparison="selective minus strict: poison_admitted_share",
                              **_paired(st.poison_admitted_share.values, sel(pc, "selective", b).poison_admitted_share.values)))
    tdf = pd.DataFrame(tests)
    pv = tdf.p.values.copy(); mask = ~np.isnan(pv); adj = np.full_like(pv, np.nan); adj[mask] = _holm(pv[mask]); tdf["holm_p"] = adj
    tdf.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)
    fam_df.to_csv(os.path.join(out_dir, "evidence_by_family.csv"), index=False)
    # ---- report
    def table(pc):
        d = summ[summ.poison == pc].copy()
        d["verification"] = d.apply(lambda r: "accept-all" if r["mode"] == "none" else f"{r['mode']} / {r['control']} @ {int(round(r['budget'] * 100))}%", axis=1)
        d["poison admitted"] = d.apply(lambda r: f"{int(r.poison_admitted)}/{int(r.n_poison)}", axis=1)
        d["rejected"] = d.poison_rejected.astype(int); d["deferred"] = d.poison_deferred.astype(int); d["fast"] = d.poison_fast.astype(int)
        d = d.sort_values(["mode", "control", "budget"])
        d = d.rename(columns={"retention": "clean retained", "correct_ret": "correct-label retained", "mislab_adm": "mislabelled admitted", "f1": "macro-F1", "asr": "backdoor ASR"})
        cols = ["verification", "poison admitted", "rejected", "deferred", "fast", "clean retained", "correct-label retained", "mislabelled admitted", "macro-F1", "backdoor ASR"]
        return _md(d[cols])
    md = [f"# ExVCL web-lab experiment\n",
          f"- stream {n_events} requests in chunks of {chunk}; {seeds} seeds; label-flip rate {poison_rate}, backdoor rate {backdoor_rate}; learner `{learner}`; budgets {list(budgets)}",
          f"- upstream text model (char n-gram TF-IDF + logistic regression, trained on families {list(upstream_attack_families)} + all benign): "
          f"accuracy {up_eval['accuracy']:.3f}, attack recall {up_eval['attack_recall']:.3f}, benign false positives {up_eval['benign_false_positive']:.3f}",
          f"- evidence: request executed in a strace sandbox ({sandbox_s:.2f} s per execution); evidence model fitted on {int(ok_cal.sum())} labelled calibration executions (the nuisance count syscall_total is dropped)",
          f"- sandbox executions are cached in `evidence_cache.npz`; per-run data in `runs.csv`\n",
          "## Evidence quality by request family (what the evidence model says about the executed request)\n", _md(fam_df),
          "\nFamilies A1-A5 are attacks whose execution opens a sensitive path; A6 (non-listed targets) and A7 (evasions this app cannot decode) look benign in the evidence; "
          "B5 (`docs/etc/passwd-*.md`) looks malicious although benign.\n"]
    for pc in ("clean", "labelflip", "backdoor"):
        md += [f"## Poison condition: {pc}\n", "`poison admitted` counts injected poison only; `mislabelled admitted` includes the upstream model's own mistakes.\n", table(pc), ""]
    md += ["## Paired tests over seeds (comparison = second minus first; Holm over all tests)\n", _md(tdf[["poison", "budget", "comparison", "mean", "lo", "hi", "p", "holm_p"]], 3)]
    with open(os.path.join(out_dir, "weblab_report.md"), "w") as f:
        f.write("\n".join(md))
    meta = dict(n_events=n_events, seeds=seeds, budgets=list(budgets), poison_rate=poison_rate, backdoor_rate=backdoor_rate, chunk=chunk, upstream=up_eval, sandbox_seconds_per_execution=sandbox_s,
                evidence_status=ev.status, runtime_s=round(time.time() - t_start, 1))
    with open(os.path.join(out_dir, "weblab_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    progress(f"done in {meta['runtime_s']} s; report: {os.path.join(out_dir, 'weblab_report.md')}")
    return dict(runs=runs, summary=summ, tests=tdf, families=fam_df, meta=meta)

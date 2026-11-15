"""evcl.evaluation -- metrics and statistics. Ground truth is used ONLY here (runner side), never by pipelines.

Operational definitions (n = counts over the whole stream unless stated otherwise):
  PAR       poisoned observations admitted / poisoned observations                                   [NaN if none]
  FRR       clean observations NOT admitted / clean observations (rejected OR deferred)
  youden    (1-PAR) - FRR   0 = no better than random throttling; >0 selective; <0 anti-selective
  selectivity PAR/(1-FRR)   1 = random; <1 selective
  accept/reject/defer_frac  admitted / (not admitted and not deferred) / deferred(budget overflow), over ALL observations
  poison_verified_frac      poisoned observations that received behavioural verification / poisoned   (verification coverage)
  clean_verified_frac       same for clean observations;  audit_coverage = audited fast-path / fast-path observations
  budget_used_frac          verifications / (budget x chunks)  [NaN if budget unlimited]; saturated_chunks = chunks with deferral
  unverified_admit          admitted without decisive behavioural evidence / admitted
  VER                       evasive poisoned admitted / evasive poisoned
  ASR_all   triggered malicious TEST samples classified benign / all malicious test samples of seen classes
  ASR_cond  same, restricted to malicious test samples the SAME model classifies as attack WITHOUT the trigger
            (removes natural misses; denominator n_cond reported)
  FPR / FNR benign test samples predicted attack / attack test samples predicted benign (binary view, seen classes)
  balanced_acc  mean per-class recall over seen classes;  macro_f1 over seen classes; f1_active over currently-active classes
  CFR       mean over DEPARTED attack classes of [recall at last active chunk - final recall], both measured on the
            test set of the regime in which the class was last active (so concept drift is not counted as forgetting)
  adapt_*   (drift only) pre-drift f1_active = mean of 3 chunks before the drift; adapt_chunks = first k>=0 with
            f1_active(drift+k) >= 0.9 x pre-drift level (NaN + censored=True if never within the stream)
  PI        clean-run minus attacked-run headline macro-F1, paired by (variant, pipeline, seed)
Metrics that can be gamed by rejecting/deferring everything: PAR, ASR, PI, VER (all -> 0 for a frozen model). Always
read them with FRR, headline macro-F1/balanced accuracy, novel-class recall, and the B0/BR controls."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .data import KIND_NAMES

_REG = {"pre": "dev_pre", "post": "dev_post"}


def active_classes(meta, t):
    return [c for (c, s, e) in meta["schedule"] if s <= t < e]


def regime_key(meta, t):
    d = meta.get("drift_chunk", -1)
    return "dev_post" if d >= 0 and t >= d else "dev_pre"


def evaluate_on(learner, X, y, meta, seen, active) -> Dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    m = np.isin(y, seen)
    X, y = X[m], y[m]
    pred = learner.predict(X)
    p, r, _, _ = precision_recall_fscore_support(y, pred, labels=seen, zero_division=0)
    ma = np.isin(y, active)
    out = dict(macro_f1=float(f1_score(y, pred, labels=seen, average="macro", zero_division=0)),
               f1_active=float(f1_score(y[ma], pred[ma], labels=active, average="macro", zero_division=0)) if ma.any() else float("nan"),
               accuracy=float(np.mean(pred == y)), balanced_acc=float(np.mean(r)),
               recall={int(c): float(v) for c, v in zip(seen, r)}, precision={int(c): float(v) for c, v in zip(seen, p)},
               fpr=float(np.mean(pred[y == 0] != 0)) if (y == 0).any() else float("nan"),
               fnr=float(np.mean(pred[y > 0] == 0)) if (y > 0).any() else float("nan"),
               asr=float("nan"), asr_cond=float("nan"), n_mal=0, n_cond=0, miss_clean=float("nan"))
    mal = y > 0
    if mal.any():
        pm = pred[mal]
        out["miss_clean"] = float(np.mean(pm == 0))
        out["n_mal"] = int(mal.sum())
        trig = meta.get("eval_trigger")
        if trig:
            Xt = X[mal].copy()
            Xt[:, trig["idx"]] = np.asarray(trig["vals"])
            pt = learner.predict(Xt)
            out["asr"] = float(np.mean(pt == 0))                         # ASR_all
            cond = pm > 0                                               # detected as attack without the trigger
            out["n_cond"] = int(cond.sum())
            if cond.any():
                out["asr_cond"] = float(np.mean(pt[cond] == 0))        # ASR_cond
    return out


def evaluate_model(learner, stream, t: int) -> Dict:
    meta = stream.meta
    first = meta["class_first_chunk"]
    seen = [c for c in sorted(first) if first[c] <= t]
    act = active_classes(meta, t)
    key = regime_key(meta, t)
    X, y = stream.eval_sets[key]
    out = evaluate_on(learner, X, y, meta, seen, act)
    out["t"], out["set"] = t, key
    by = {key: out["recall"]}
    other = "dev_post" if key == "dev_pre" else "dev_pre"
    Xo, yo = stream.eval_sets[other]
    mo = np.isin(yo, seen)
    po = learner.predict(Xo[mo])
    by[other] = {int(c): float(np.mean(po[yo[mo] == c] == c)) for c in seen}
    out["recall_by_set"] = by
    return out


def evaluate_final(learner, stream) -> Dict:
    """Headline evaluation on the FINAL test period (only exists in final mode); computed once at stream end."""
    meta = stream.meta
    t = len(stream.chunks) - 1
    first = meta["class_first_chunk"]
    seen = [c for c in sorted(first) if first[c] <= t]
    X, y = stream.eval_sets["final"]
    out = evaluate_on(learner, X, y, meta, seen, active_classes(meta, t))
    out["t"], out["set"] = t, "final"
    return out


def new_accumulator() -> Dict:
    return dict(n_obs=0, n_chunks=0, poison_total=0, poison_admitted=0, poison_verified=0, admitted_total=0,
                admitted_poison=0, admitted_label_wrong=0, admitted_unverified=0, clean_total=0, clean_rejected=0,
                clean_verified=0, evasion_total=0, evasion_admitted=0, n_verified=0, n_audit=0, n_deferred=0,
                fast_total=0, exec_failed=0, saturated_chunks=0, kind_total={}, kind_admitted={}, poison_ids=set(),
                t_triage=0.0, t_sandbox=0.0, t_fusion=0.0, t_admission=0.0, t_learn=0.0, t_process=0.0,
                cpu_s=0.0, evals=[])


def update_accumulator(acc, chunk, obs, out, wall, cpu):
    """chunk = full ground-truth Chunk (runner only); obs = what the pipeline saw; out = pipeline decision."""
    adm, p, route = out.admit, chunk.is_poison, out.route
    ver = route == 1
    acc["n_obs"] += len(chunk)
    acc["n_chunks"] += 1
    acc["poison_total"] += int(p.sum())
    acc["poison_admitted"] += int((p & adm).sum())
    acc["poison_verified"] += int((p & ver).sum())
    acc["admitted_total"] += int(adm.sum())
    acc["admitted_poison"] += int((adm & p).sum())
    acc["admitted_label_wrong"] += int((adm & (out.label != chunk.y_true)).sum())
    acc["admitted_unverified"] += int((adm & ~ver).sum())
    acc["clean_total"] += int((~p).sum())
    acc["clean_rejected"] += int(((~p) & (~adm)).sum())
    acc["clean_verified"] += int(((~p) & ver).sum())
    ev = chunk.evasive & p
    acc["evasion_total"] += int(ev.sum())
    acc["evasion_admitted"] += int((ev & adm).sum())
    acc["fast_total"] += int((route == 0).sum())
    acc["exec_failed"] += int(out.n_exec_failed)
    acc["saturated_chunks"] += int(out.n_deferred > 0)
    acc["poison_ids"].update(int(i) for i in obs.obs_id[p])
    for k in np.unique(chunk.kind[p]):
        mk = p & (chunk.kind == k)
        acc["kind_total"][int(k)] = acc["kind_total"].get(int(k), 0) + int(mk.sum())
        acc["kind_admitted"][int(k)] = acc["kind_admitted"].get(int(k), 0) + int((mk & adm).sum())
    acc["n_verified"] += out.n_verified
    acc["n_audit"] += out.n_audit
    acc["n_deferred"] += out.n_deferred
    for s in ("triage", "sandbox", "fusion", "admission", "learn"):
        acc["t_" + s] += out.timings.get(s, 0.0)
    acc["t_process"] += wall
    acc["cpu_s"] += cpu


def _ratio(a, b):
    return float(a) / b if b else float("nan")


def derive_rates(acc: Dict) -> Dict:
    par = _ratio(acc["poison_admitted"], acc["poison_total"])
    frr = _ratio(acc["clean_rejected"], acc["clean_total"])
    ok = not np.isnan(par) and not np.isnan(frr)
    return dict(
        PAR=par, FRR=frr, youden=(1 - par) - frr if ok else float("nan"),
        selectivity=_ratio(par, 1 - frr) if ok else float("nan"),
        admitted_poison_frac=_ratio(acc["admitted_poison"], acc["admitted_total"]),
        label_error_admitted=_ratio(acc["admitted_label_wrong"], acc["admitted_total"]),
        unverified_admit=_ratio(acc["admitted_unverified"], acc["admitted_total"]),
        VER=_ratio(acc["evasion_admitted"], acc["evasion_total"]))


def forgetting_departed(evals: List[Dict], last_chunk: Dict[int, int], n_chunks: int, drift_chunk: int = -1):
    """Mean over departed attack classes of [recall at last active chunk - final recall]; both measured on the
    test set of the regime in which the class was last active (drift is not counted as forgetting)."""
    final = evals[-1]
    vals = []
    for c, end in last_chunk.items():
        if c == 0 or end >= n_chunks or c not in final["recall_by_set"].get("dev_pre", {}):
            continue
        key = "dev_post" if drift_chunk >= 0 and (end - 1) >= drift_chunk else "dev_pre"
        before = [e for e in evals if e["t"] <= end - 1 and c in e["recall_by_set"].get(key, {})]
        if before:
            vals.append(before[-1]["recall_by_set"][key][c] - final["recall_by_set"][key][c])
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def adaptation_metrics(evals: List[Dict], drift_chunk: int, frac=0.9, window=3) -> Dict:
    nan = float("nan")
    out = dict(adapt_pre_f1=nan, adapt_drop=nan, adapt_chunks=nan, adapt_censored=False, adapt_auc5_rel=nan)
    if drift_chunk < 1:
        return out
    pre = [e["f1_active"] for e in evals if drift_chunk - window <= e["t"] < drift_chunk and not np.isnan(e["f1_active"])]
    post = [e for e in evals if e["t"] >= drift_chunk]
    if not pre or not post:
        return out
    L = float(np.mean(pre))
    out.update(adapt_pre_f1=L, adapt_drop=L - post[0]["f1_active"],
               adapt_auc5_rel=float(np.mean([e["f1_active"] for e in post[:5]]) / L) if L > 0 else nan)
    rec = next((k for k, e in enumerate(post) if e["f1_active"] >= frac * L), None)
    out["adapt_chunks"], out["adapt_censored"] = (float(rec) if rec is not None else nan), rec is None
    return out


def finalize_metrics(acc: Dict, meta: Dict, learner, n_chunks: int, telemetry_source: str,
                     final_eval: Dict | None = None, budget: int | None = None) -> Dict:
    ev = acc["evals"]
    base = set(meta["base_classes"])
    first = meta["class_first_chunk"]
    last = ev[-1]
    cfr, n_dep = forgetting_departed(ev, meta["class_last_chunk"], n_chunks, meta.get("drift_chunk", -1))
    novel = [c for c in last["recall"] if c not in base and c > 0]
    plus3 = [h[0]["recall"][c] for c in novel
             for h in [[e for e in ev if e["t"] == first[c] + 3 and c in e["recall"]]] if h]
    n = max(acc["n_obs"], 1)
    ids = learner.buffer.ids()
    contam = float(np.mean([int(i) in acc["poison_ids"] for i in ids])) if len(ids) else 0.0
    simulated = telemetry_source == "SIMULATED"
    pipe_only = acc["t_process"] - acc["t_sandbox"]
    head = final_eval if final_eval is not None else last
    res = dict(
        final_macro_f1=last["macro_f1"], mean_macro_f1=float(np.mean([e["macro_f1"] for e in ev])),
        asr_final=last["asr"], miss_clean_final=last["miss_clean"],
        headline_split="final" if final_eval is not None else "dev",
        headline_macro_f1=head["macro_f1"], headline_balanced_acc=head["balanced_acc"], headline_fpr=head["fpr"],
        headline_fnr=head["fnr"], headline_asr_all=head["asr"], headline_asr_cond=head["asr_cond"],
        headline_n_mal=head["n_mal"], headline_n_cond=head["n_cond"], headline_f1_active=head["f1_active"],
        CFR=cfr, CFR_n_departed_classes=n_dep,
        novel_recall_final=float(np.mean([head["recall"][c] for c in novel if c in head["recall"]])) if novel else float("nan"),
        novel_recall_intro_plus3=float(np.mean(plus3)) if plus3 else float("nan"),
        buffer_contamination=contam, n_obs=acc["n_obs"], poison_total=acc["poison_total"],
        accept_frac=_ratio(acc["admitted_total"], n), defer_frac=_ratio(acc["n_deferred"], n),
        reject_frac=_ratio(acc["n_obs"] - acc["admitted_total"] - acc["n_deferred"], n),
        verified_frac=_ratio(acc["n_verified"], n), deferred_frac=_ratio(acc["n_deferred"], n),
        audit_frac=_ratio(acc["n_audit"], n), audit_coverage=_ratio(acc["n_audit"], acc["fast_total"]),
        poison_verified_frac=_ratio(acc["poison_verified"], acc["poison_total"]),
        clean_verified_frac=_ratio(acc["clean_verified"], acc["clean_total"]),
        exec_failed_frac=_ratio(acc["exec_failed"], acc["n_verified"]),
        budget_used_frac=_ratio(acc["n_verified"], budget * acc["n_chunks"]) if budget is not None and budget < 10 ** 5 else float("nan"),
        saturated_chunk_frac=_ratio(acc["saturated_chunks"], acc["n_chunks"]),
        lat_triage_userspace_us=1e6 * acc["t_triage"] / n, lat_fusion_userspace_us=1e6 * acc["t_fusion"] / n,
        lat_admission_userspace_us=1e6 * acc["t_admission"] / n, lat_learn_userspace_us=1e6 * acc["t_learn"] / n,
        lat_sandbox_s_per_exec=float("nan") if (simulated or telemetry_source == "NONE")
        else _ratio(acc["t_sandbox"], acc["n_verified"]),                     # NaN unless a REAL adapter ran
        throughput_pipeline_only_obs_per_s=_ratio(acc["n_obs"], pipe_only),   # excludes sandbox time
        throughput_end_to_end_obs_per_s=float("nan") if simulated else _ratio(acc["n_obs"], acc["t_process"]),
        cpu_s=acc["cpu_s"], memory_mb=float("nan"))                            # memory: NOT measured (shared process)
    res.update(derive_rates(acc))
    res.update(adaptation_metrics(ev, meta.get("drift_chunk", -1)))
    for k, v in acc["kind_total"].items():
        res[f"PAR_{KIND_NAMES[k]}"] = _ratio(acc["kind_admitted"].get(k, 0), v)
    for c, v in head["recall"].items():
        res[f"recall_c{c}"] = v
        res[f"precision_c{c}"] = head["precision"].get(c, float("nan"))
    return res


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    means = r.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2 or np.allclose(a, b):
        return float("nan") if len(a) < 2 else 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def paired_stats(a, b, seed=0) -> Dict:
    """Paired summary over INDEPENDENT RUNS (seeds): mean diff, SD, bootstrap CI, d_z, Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    n = int(m.sum())
    if n < 2:
        return dict(n=n, mean_diff=float("nan"), sd_diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    d_z=float("nan"), p_wilcoxon=float("nan"))
    mean, lo, hi = bootstrap_ci(d, seed=seed)
    sd = float(d.std(ddof=1))
    return dict(n=n, mean_diff=mean, sd_diff=sd, ci_lo=lo, ci_hi=hi, d_z=float(mean / sd) if sd > 0 else float("nan"),
                p_wilcoxon=paired_wilcoxon(a, b))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.sum(~np.isnan(p)))
    out = np.full(len(p), np.nan)
    run = 0.0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def add_pi(df):
    """PI = clean-run minus attacked-run HEADLINE macro-F1 (final test period in final mode), paired by seed."""
    keys = ["variant", "pipeline", "seed"]
    cols = ["headline_macro_f1", "headline_balanced_acc", "headline_asr_all", "headline_asr_cond", "headline_fnr"]
    clean = df[df.attack == "none"][keys + cols]
    clean = clean.rename(columns={c: c + "_clean" for c in cols})
    m = df.merge(clean, on=keys, how="left")
    m["PI_macro_f1"] = m.headline_macro_f1_clean - m.headline_macro_f1
    m["PI_balanced_acc"] = m.headline_balanced_acc_clean - m.headline_balanced_acc
    m["delta_asr_all"] = m.headline_asr_all - m.headline_asr_all_clean
    m["delta_fnr"] = m.headline_fnr - m.headline_fnr_clean
    return m
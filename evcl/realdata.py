"""evcl.realdata -- prepare real flow CSVs (e.g. CIC-IDS2017-style CICFlowMeter exports) for the chronological loader.

Nothing here downloads data. You supply the files (check the licence and cite the dataset paper). The function:
  * strips column names, removes identifier columns from the feature set, maps +/-inf to 0;
  * derives a monotone timestamp `ts`: day index from the file name (Monday..Sunday) + time-of-day parsed from the time column
    (avoids day/month ambiguity); falls back to row order inside each file and records that fact in the report;
  * drops classes with too few rows and thins large classes UNIFORMLY IN TIME (chronology preserved);
  * writes one CSV with feature columns + `label` + `ts` (+ `entity`), and returns a report dict.
It does NOT fix label errors. For CIC-IDS2017/2018 prefer a corrected re-labelled release and cite it."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

_DAYS = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare_flow_csv(files: Iterable[str], out_csv: str, label_col: str = "Label", time_col: Optional[str] = "Timestamp",
                     benign_label: str = "BENIGN", entity_col: Optional[str] = None,
                     drop_cols: Iterable[str] = ("Flow ID", "Source IP", "Destination IP", "Source Port", "SimillarHTTP", "Unnamed: 0",
                                                 "id", "Src IP", "Dst IP", "Src Port", "Attempted Category"),
                     label_map: Optional[Dict[str, str]] = None, min_rows_per_class: int = 400,
                     max_rows_per_class: int = 20000, day_from_filename: bool = True, seed: int = 0, fix_12h: bool = True,
                     log_features: bool = False, keep_classes: Optional[Iterable[str]] = None,
                     drop_attempted: bool = True) -> Dict:
    import pandas as pd
    files = list(files)
    if not files:
        raise ValueError("no input files")
    lc = lambda s: str(s).strip().lower()  # noqa: E731
    parts, report = [], dict(files={}, time_source=set(), dropped_classes={}, benign_label=benign_label, time_12h_fix_files=[], attempted_dropped={})
    for order, path in enumerate(files):
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:                      # some CIC-IDS2017 exports contain a cp1252/latin-1 byte inside label text
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
            report.setdefault("encoding_fallback", []).append(os.path.basename(path))
        df.columns = [str(c).strip() for c in df.columns]
        cols = {lc(c): c for c in df.columns}
        if lc(label_col) not in cols:
            raise KeyError(f"{os.path.basename(path)}: label column {label_col!r} not found; columns start with {list(df.columns)[:8]}")
        # GLF exports contain rows with an empty Label (and empty features). Left in, they become a bogus "nan" attack class.
        _l = df[cols[lc(label_col)]]
        _blank = _l.isna() | _l.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
        _skipc = {lc(label_col), lc(time_col or ""), lc(entity_col or "")} | {lc(c) for c in drop_cols}
        _fc = [c for c in df.columns if lc(c) not in _skipc]
        _blank |= df[_fc].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
        if drop_attempted:
            _att = _l.astype(str).str.lower().str.contains("attempted", na=False)
            _ac = [c for c in df.columns if lc(c) == "attempted category"]
            if _ac:
                _att |= (pd.to_numeric(df[_ac[0]], errors="coerce").fillna(-1) != -1)
            report.setdefault("attempted_dropped", {})[os.path.basename(path)] = int((_att & ~_blank).sum())
            _blank |= _att
        n_blank = int(_blank.sum())
        if n_blank:
            df = df[~_blank].reset_index(drop=True)
        lab = (df[cols[lc(label_col)]].astype(str).str.replace(r"[^\x20-\x7E]+", "-", regex=True)   # non-ASCII (e.g. en-dash byte) -> "-"
               .str.replace(r"\s*-\s*", " - ", regex=True).str.strip())
        if label_map:
            lab = lab.map(lambda x: label_map.get(x, x))
        lab = lab.where(lab.str.lower() != benign_label.lower(), benign_label)
        day = next((v for k, v in _DAYS.items() if k in os.path.basename(path).lower()), order) if day_from_filename else 0
        n = len(df)
        if time_col and lc(time_col) in cols:
            _t = df[cols[lc(time_col)]].astype(str).str.strip()
            parsed = pd.to_datetime(_t, errors="coerce", dayfirst=True)
            _bad = parsed.isna() & df[cols[lc(time_col)]].notna()
            if _bad.any():                                  # mixed formats within one file: parse the failed rows one by one
                try:
                    parsed[_bad] = pd.to_datetime(_t[_bad], errors="coerce", dayfirst=True, format="mixed")
                except (TypeError, ValueError):
                    pass
            t_unparsed_frac = float(parsed.isna().mean())
            hrs = parsed.dt.hour
            if fix_12h and hrs.notna().any() and hrs.max() <= 12 and hrs.between(1, 7).any():
                # CIC-IDS2017 writes clock times WITHOUT am/pm (captures ran about 08:00-17:00), so 01:00-07:59 means 13:00-19:59.
                # Left alone, afternoon flows would sort BEFORE the morning ones. Applied only when the file never shows an hour > 12.
                hrs = hrs.where(~hrs.between(1, 7), hrs + 12)
                report["time_12h_fix_files"].append(os.path.basename(path))
            tod = (hrs * 3600 + parsed.dt.minute * 60 + parsed.dt.second).to_numpy(float)
            if np.isnan(tod).mean() > 0.5:
                tod = np.linspace(0, 86399, n)
                report["time_source"].add("row-order fallback (time column unparseable)"); _tsrc = "row-order fallback"
            else:
                tod = np.where(np.isnan(tod), np.nanmedian(tod), tod)
                report["time_source"].add("file-name day + parsed time-of-day" if day_from_filename else "parsed datetime"); _tsrc = "parsed time-of-day"
            ts = (day * 86400 + tod) if day_from_filename else (parsed.astype("int64").to_numpy(float) / 1e9)
        else:
            ts = day * 86400 + np.linspace(0, 86399, n)
            report["time_source"].add("file-name day + row order (NO time column)"); _tsrc = "row order (no time column)"
        ent = df[cols[lc(entity_col)]].astype(str).to_numpy() if entity_col and lc(entity_col) in cols else None
        skip = {lc(label_col), lc(time_col or ""), lc(entity_col or "")} | {lc(c) for c in drop_cols}
        feats = df[[c for c in df.columns if lc(c) not in skip]].apply(pd.to_numeric, errors="coerce")
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats["label"], feats["ts"] = lab.to_numpy(), ts
        if ent is not None:
            feats["entity"] = ent
        parts.append(feats)
        report["files"][os.path.basename(path)] = dict(rows=n, rows_dropped_blank=n_blank, sha256=_sha(path), day_index=int(day),
                                                      time_source=_tsrc, time_unparsed_frac=(t_unparsed_frac if time_col and lc(time_col) in cols else None),
                                                      sample_timestamps=(df[cols[lc(time_col)]].astype(str).head(3).tolist() if time_col and lc(time_col) in cols else None))
    cols0 = [set(p.columns) for p in parts]
    common = set.intersection(*cols0)
    parts = [p[[c for c in parts[0].columns if c in common]] for p in parts]
    df = pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    counts = df.label.value_counts()
    # name resolution: the improved/corrected releases spell classes differently ("Benign", "FTP Patator", "Botnet Ares"). A requested name matches an
    # available label by normalised spelling; otherwise by a UNIQUE label starting with it; anything ambiguous or absent is an error (no silent substitution).
    nz = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())  # noqa: E731
    avail = [str(k) for k in counts.index]

    def _resolve(name):
        exact = [a for a in avail if nz(a) == nz(name)]
        if len(exact) == 1:
            return exact[0]
        starts = [a for a in avail if nz(a).startswith(nz(name))]
        if not exact and len(starts) == 1:
            return starts[0]
        raise ValueError(f"cannot resolve class {name!r}: candidates {exact or starts or 'none'} | available labels (rows): {dict(counts.astype(int))}")
    mapping = {}
    try:
        mapping[_resolve(benign_label)] = benign_label
    except ValueError:
        pass
    for c in (keep_classes or []):
        c = str(c).strip()
        if c != benign_label:
            mapping[_resolve(c)] = c
    mapping = {k: v for k, v in mapping.items() if k != v}
    if mapping:
        df["label"] = df["label"].replace(mapping)
        counts = df.label.value_counts()
        report["class_mapping"] = mapping
    report["class_counts_raw"] = {str(k): int(v) for k, v in counts.items()}
    keep = [c for c, v in counts.items() if v >= min_rows_per_class]
    if keep_classes:                       # explicit, pre-specified class set (benign is always kept); no silent substitutions
        want = [benign_label] + [str(c).strip() for c in keep_classes if str(c).strip() != benign_label]
        missing = [c for c in want if c not in keep]
        if missing:
            raise ValueError("requested classes missing or below min_rows_per_class=%d: %s | available (rows): %s"
                             % (min_rows_per_class, {c: int(counts.get(c, 0)) for c in missing}, {str(k): int(v) for k, v in counts.items()}))
        keep = want
        report["keep_classes"] = want
    if benign_label not in keep:
        raise ValueError(f"benign class {benign_label!r} missing or too small")
    report["dropped_classes"] = {str(c): int(v) for c, v in counts.items() if c not in keep}
    df = df[df.label.isin(keep)]
    out = []
    for c, g in df.groupby("label", sort=False):
        if len(g) > max_rows_per_class:
            idx = np.unique(np.linspace(0, len(g) - 1, max_rows_per_class).round().astype(int))   # uniform thinning in time
            g = g.iloc[idx]
        out.append(g)
    df = pd.concat(out).sort_values("ts", kind="stable").reset_index(drop=True)
    if log_features:                       # signed log1p: tames the heavy tails of rate/byte/duration features; identical for every method
        fcols = [c for c in df.columns if c not in ("label", "ts", "entity")]
        X = df[fcols].to_numpy(float)
        df[fcols] = np.sign(X) * np.log1p(np.abs(X))
    report["log_features"] = bool(log_features)
    df.to_csv(out_csv, index=False)
    report.update(time_source=sorted(report["time_source"]), encoding_fallback=report.get("encoding_fallback", []), time_12h_fix_files=report["time_12h_fix_files"], class_counts_final={str(k): int(v) for k, v in df.label.value_counts().items()},
                  n_features=int(df.shape[1] - 2 - ("entity" in df.columns)), out_csv=out_csv, out_sha256=_sha(out_csv),
                  note="labels are NOT corrected here; use a corrected release for CIC-IDS2017/2018 and cite it")
    return report
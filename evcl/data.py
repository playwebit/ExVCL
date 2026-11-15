"""evcl.data -- observation chunks, defender-visible views, class-incremental streams, dataset providers.

Information barriers (enforced by types + tests):
  * Chunk            : full record incl. ground truth. Used ONLY by the experiment runner / attacks / metrics.
  * ObservedChunk    : what a defender can see (features, labelling-channel label, opaque position-based id).
  * DefenderView     : what a defender gets at setup (trusted bootstrap data, dimensions). No test data, no meta.
  * OracleChannel    : SIMULATION-ONLY ground-truth side channel, consumed solely by SimulatedSandbox.

The synthetic generator exists ONLY to validate pipeline logic. It is not a network-traffic dataset."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from .core import DataCfg, make_rng

KIND = {"none": 0, "label": 1, "clean_label": 2, "replay_flood": 3, "backdoor": 4, "slow": 5, "evasion": 6,
        "replay_duplicate": 7, "triage_aware": 8}
KIND_NAMES = {v: k for k, v in KIND.items()}
_FIELDS = ("X", "y_obs", "y_true", "is_poison", "kind", "evasive", "trigger", "uid", "evasion")
EVASION_MODES = {"env": 1, "delayed": 2, "mimicry": 3, "partial": 4}
CHUNK_ID_STRIDE = 1_000_000


@dataclass(frozen=True)
class ObservedChunk:
    X: np.ndarray
    y_obs: np.ndarray        # label supplied by the (attackable) labelling channel
    obs_id: np.ndarray       # opaque: chunk_index * STRIDE + position (position is shuffled; carries no poison info)

    def __len__(self):
        return len(self.y_obs)


@dataclass(frozen=True)
class DefenderView:
    """Everything a defender may know at setup: trusted bootstrap period + trusted CALIBRATION period (both
    labelled by a trusted source). No class count, no stream, no evaluation data."""
    n_features: int
    bootstrap_X: np.ndarray
    bootstrap_y: np.ndarray
    calib_X: np.ndarray
    calib_y: np.ndarray


class OracleChannel:
    """SIMULATION-ONLY: maps obs_id -> ground-truth class / evasion latents. Real adapters never receive this.
    Simulated telemetry is generated FROM these latents, hence it is not independent of ground truth."""

    def __init__(self, ids, classes, evasive, evasion=None):
        classes = np.asarray(classes)
        evasion = np.zeros(len(classes), np.int8) if evasion is None else evasion
        self._m = {int(i): (int(c), bool(e), int(m)) for i, c, e, m in zip(ids, classes, evasive, evasion)}

    def lookup(self, obs_id):                      # v1 interface: (true_malicious, evasive)
        c, e, _ = self._m[int(obs_id)]
        return c > 0, e

    def lookup_full(self, obs_id):                 # v2 interface: (true_class, evasive, evasion_mode)
        return self._m[int(obs_id)]


@dataclass
class Chunk:
    X: np.ndarray
    y_obs: np.ndarray
    y_true: np.ndarray       # ground truth (evaluation / simulation only)
    is_poison: np.ndarray    # attacker touched this observation (metric bookkeeping only)
    kind: np.ndarray
    evasive: np.ndarray      # latent: malicious behaviour tries to evade the sandbox
    trigger: np.ndarray
    uid: np.ndarray          # generation-order id (leaks poison status via ordering -> never shown to defenders)
    evasion: np.ndarray      # latent evasion mode (0 none; see EVASION_MODES); simulation/evaluation only

    def __len__(self):
        return len(self.y_obs)

    def copy(self) -> "Chunk":
        return Chunk(**{f: getattr(self, f).copy() for f in _FIELDS})

    def take(self, idx) -> "Chunk":
        return Chunk(**{f: getattr(self, f)[idx] for f in _FIELDS})

    def append(self, o: "Chunk") -> "Chunk":
        return Chunk(**{f: np.concatenate([getattr(self, f), getattr(o, f)]) for f in _FIELDS})

    def obs_ids(self, t: int) -> np.ndarray:
        return t * CHUNK_ID_STRIDE + np.arange(len(self), dtype=np.int64)

    def observed(self, t: int) -> ObservedChunk:
        X = self.X.copy()
        X.setflags(write=False)
        y = self.y_obs.copy()
        y.setflags(write=False)
        return ObservedChunk(X, y, self.obs_ids(t))

    def oracle(self, t: int) -> OracleChannel:
        return OracleChannel(self.obs_ids(t), self.y_true, self.evasive, self.evasion)


@dataclass
class StreamData:
    bootstrap_X: np.ndarray
    bootstrap_y: np.ndarray
    calib_X: np.ndarray
    calib_y: np.ndarray
    chunks: List[Chunk]
    eval_sets: Dict[str, Any]          # 'dev_pre', 'dev_post' always; 'final' ONLY in final mode
    n_classes: int                     # evaluator/config side only; never given to defenders
    n_features: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_X(self):                  # legacy alias (development test, pre-drift regime)
        return self.eval_sets["dev_pre"][0]

    @property
    def test_y(self):
        return self.eval_sets["dev_pre"][1]

    def freeze(self):
        for a in (self.bootstrap_X, self.bootstrap_y, self.calib_X, self.calib_y):
            a.setflags(write=False)
        for X, y in self.eval_sets.values():
            X.setflags(write=False)
            y.setflags(write=False)
        return self

    def defender_view(self) -> DefenderView:
        return DefenderView(self.n_features, self.bootstrap_X, self.bootstrap_y, self.calib_X, self.calib_y)

    def eval_fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.bootstrap_X, self.bootstrap_y, self.calib_X, self.calib_y):
            h.update(np.ascontiguousarray(a).tobytes())
        for k in sorted(self.eval_sets):
            h.update(k.encode())
            for a in self.eval_sets[k]:
                h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.eval_fingerprint().encode())
        for c in self.chunks:
            for f in _FIELDS:
                h.update(np.ascontiguousarray(getattr(c, f)).tobytes())
        return h.hexdigest()[:16]


def row_hashes(X: np.ndarray) -> set:
    X = np.ascontiguousarray(X, dtype=np.float64)
    return {hashlib.blake2b(r.tobytes(), digest_size=12).digest() for r in X}


def assert_disjoint_rows(A: np.ndarray, B: np.ndarray, what: str):
    n = len(row_hashes(A) & row_hashes(B))
    if n:
        raise AssertionError(f"LEAKAGE: {n} identical feature rows shared between {what}")


def _active(schedule, t):
    return [c for (c, s, e) in schedule if s <= t < e]


def chunk_counts(cfg: DataCfg, schedule, t) -> Dict[int, int]:
    active = _active(schedule, t)
    atk = [c for c in active if c > 0]
    counts: Dict[int, int] = {}
    if atk:
        n_b = int(round(cfg.benign_frac * cfg.chunk_size))
        rest = cfg.chunk_size - n_b
        base = rest // len(atk)
        for i, c in enumerate(atk):
            counts[c] = base + (1 if i < rest - base * len(atk) else 0)
        counts[0] = n_b
    else:
        counts[0] = cfg.chunk_size
    return counts


def _build_chunks(cfg: DataCfg, sample: Callable, rng, schedule):
    drift_on = cfg.drift_chunk >= 0
    chunks, uid = [], 0
    for t in range(cfg.n_chunks):
        counts = chunk_counts(cfg, schedule, t)
        Xs, ys = [], []
        for c in sorted(counts):
            Xs.append(sample(c, counts[c], rng, "post" if drift_on and t >= cfg.drift_chunk else "pre"))
            ys.append(np.full(counts[c], c, dtype=np.int64))
        X, y = np.vstack(Xs), np.concatenate(ys)
        p = rng.permutation(len(y))
        X, y = X[p], y[p]
        n = len(y)
        chunks.append(Chunk(X=X, y_obs=y.copy(), y_true=y.copy(), is_poison=np.zeros(n, bool),
                            kind=np.zeros(n, np.int8), evasive=np.zeros(n, bool), trigger=np.zeros(n, bool),
                            uid=np.arange(uid, uid + n, dtype=np.int64), evasion=np.zeros(n, np.int8)))
        uid += n
    return chunks, uid


def _apply_label_noise(chunks, rate, K, rng):
    """Natural labelling noise (NOT attacker action): y_obs changed to a different class; is_poison stays False."""
    if rate <= 0:
        return
    for ch in chunks:
        m = rng.random(len(ch)) < rate
        ch.y_obs[m] = (ch.y_true[m] + rng.integers(1, K, size=int(m.sum()))) % K


def _synthetic_sampler(cfg: DataCfg, rng, seed: int):
    d, K = cfg.n_features, cfg.n_classes
    b_means = rng.normal(0, 0.5, size=(3, d))
    centre = b_means.mean(0)
    dirs = rng.normal(size=(K - 1, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    a_means = centre + cfg.class_sep * dirs
    scales = rng.uniform(0.8, 1.3, size=K)

    delta = np.zeros(d)
    if cfg.drift_chunk >= 0:                       # separate RNG stream: legacy draws are unaffected when drift is off
        dr = make_rng(seed, "drift").normal(size=d)
        delta = cfg.drift_norm * dr / np.linalg.norm(dr)

    def sample(c, n, r, regime="pre"):
        shift = delta if regime == "post" else 0.0
        if c == 0:
            comp = r.integers(0, 3, size=n)
            return b_means[comp] + shift + r.normal(size=(n, d)) * scales[0]
        return a_means[c - 1] + shift + r.normal(size=(n, d)) * scales[c]
    return sample


def _auto_schedule(cfg: DataCfg, K: int):
    return tuple([(c, 0, cfg.n_chunks) for c in cfg.base_classes if c < K] +
                 [(c, min(6 * (c - 2), cfg.n_chunks - 1), cfg.n_chunks) for c in range(3, K)])


_SEG = {"boot": 0, "calib": 1, "stream": 2, "dev": 3, "final": 4}


def _build_csv(cfg: DataCfg, rng, include_final: bool):
    """Real flow table. Per class (chronological, or random only with explicit opt-in) the layout is
         [bootstrap][calibration][stream demand (consumed sequentially, NO replacement)][gap][dev test][final test]
       Calibration/bootstrap contain only classes active at chunk 0 (no future labels). The layout (and therefore
       every development artefact) is IDENTICAL whether or not the final test period is materialised.
       Exact-duplicate feature rows are removed; entities (e.g. hosts/sessions) that span >1 period are PURGED."""
    import pandas as pd
    df = pd.read_csv(cfg.csv_path)
    drop = [cfg.csv_label_col] + [c for c in (cfg.csv_time_col, cfg.csv_entity_col) if c]
    lab = df[cfg.csv_label_col].astype(str)
    feats = (df.drop(columns=drop).select_dtypes("number").replace([np.inf, -np.inf], np.nan).fillna(0.0))
    tv = None
    if cfg.csv_time_col:
        col = df[cfg.csv_time_col]
        tv = (col.to_numpy(float) if pd.api.types.is_numeric_dtype(col)
              else pd.to_datetime(col, errors="coerce").astype("int64").to_numpy(float))
    ent = df[cfg.csv_entity_col].astype(str).to_numpy() if cfg.csv_entity_col else None
    n_raw = len(df)
    keep0 = ~feats.duplicated(keep="first").to_numpy()
    if tv is not None:
        keep0 &= ~np.isnan(tv)
    feats, lab = feats[keep0], lab[keep0]
    tv = tv[keep0] if tv is not None else None
    ent = ent[keep0] if ent is not None else None
    order = [cfg.csv_benign_label] + [c for c in lab.value_counts().index if c != cfg.csv_benign_label]
    order = order[: cfg.n_classes]
    ysel = lab.map({c: i for i, c in enumerate(order)})
    ok = ysel.notna().to_numpy()
    X_all, y_all = feats.to_numpy(float)[ok], ysel.to_numpy()[ok].astype(np.int64)
    t_all = tv[ok] if tv is not None else None
    e_all = ent[ok] if ent is not None else None
    K, d = len(order), X_all.shape[1]
    schedule = cfg.schedule or _auto_schedule(cfg, K)
    first0 = {c for (c, st, en) in schedule if st == 0}
    demand = {c: 0 for c in range(K)}
    for t in range(cfg.n_chunks):
        for c, n in chunk_counts(cfg, schedule, t).items():
            demand[c] += n
    n_b = {c: (cfg.bootstrap_per_class * (2 if c == 0 else 1) if c in cfg.base_classes else 0) for c in range(K)}
    n_c = {c: (cfg.calib_per_class * (2 if c == 0 else 1) if c in first0 and c in cfg.base_classes else 0) for c in range(K)}
    seg = np.full(len(X_all), -1)
    class_idx: Dict = {}
    need_of = {}
    for c in range(K):
        idx = np.flatnonzero(y_all == c)
        idx = idx[np.argsort(t_all[idx], kind="stable")] if t_all is not None else idx
        if t_all is None:
            idx = np.random.default_rng(1234 + c).permutation(idx)       # fixed layout seed: random split is dev-only
        need = dict(boot=n_b[c], calib=n_c[c], stream=demand[c], dev=cfg.test_per_class, final=cfg.final_test_per_class)
        need_of[c] = need
        tot = sum(need.values())
        if len(idx) < tot:
            raise ValueError(f"class {order[c]!r}: need {tot} unique rows (bootstrap {n_b[c]} + calibration {n_c[c]} + "
                             f"stream {demand[c]} + dev test {cfg.test_per_class} + final test "
                             f"{cfg.final_test_per_class}) but only {len(idx)} available after de-duplication; reduce sizes")
        slack = 1.0
        if e_all is not None:                                              # reserve extra rows that entity purge may remove
            slack = min(1.6, len(idx) / tot)
        sizes = {k: int(np.ceil(v * slack)) for k, v in need.items()}
        sizes["stream"] = max(sizes["stream"], need["stream"])
        while sum(sizes.values()) > len(idx):
            k = max(sizes, key=lambda z: sizes[z] - need[z])
            sizes[k] -= 1
        p = 0
        for name in ("boot", "calib", "stream"):
            seg[idx[p:p + sizes[name]]] = _SEG[name]
            class_idx[(c, name)] = idx[p:p + sizes[name]]
            p += sizes[name]
        n = len(idx)
        class_idx[(c, "final")] = idx[n - sizes["final"]:]
        class_idx[(c, "dev")] = idx[n - sizes["final"] - sizes["dev"]: n - sizes["final"]]
        seg[class_idx[(c, "final")]] = _SEG["final"]
        seg[class_idx[(c, "dev")]] = _SEG["dev"]
    purged = 0
    if e_all is not None:
        # entity assigned to the period of its FIRST appearance; its rows in later periods are purged
        first_seg = pd.Series(seg[seg >= 0]).groupby(e_all[seg >= 0]).min()
        row_first = pd.Series(e_all).map(first_seg).to_numpy()
        later = (seg >= 0) & (seg > np.nan_to_num(row_first, nan=99))
        purged = int(later.sum())
        seg[later] = -1
        for key, idx in list(class_idx.items()):
            class_idx[key] = idx[seg[idx] >= 0]
    for (c, name), idx in list(class_idx.items()):                         # trim to the required counts (earliest first)
        need = need_of[c][name]
        if len(idx) < need:
            raise ValueError(f"class {order[c]!r}: only {len(idx)} rows left in period {name!r} after entity purge "
                             f"(need {need}); reduce sizes or relax the entity column")
        class_idx[(c, name)] = idx[:need] if name != "final" and name != "dev" else idx[len(idx) - need:]
    if e_all is not None:
        used = np.concatenate([v for v in class_idx.values()])
        segs = np.concatenate([np.full(len(v), _SEG[k[1]]) for k, v in class_idx.items()])
        chk = pd.DataFrame({"e": e_all[used], "s": segs}).groupby("e").s.nunique()
        assert chk.max() == 1, "entity leakage across periods"
    pools = {c: X_all[class_idx[(c, "stream")]] for c in range(K)}
    cursor = {c: 0 for c in range(K)}

    def sample(c, n, r, regime="pre"):
        i = cursor[c]
        cursor[c] = i + n
        return pools[c][i:i + n]
    boot = [(c, X_all[class_idx[(c, "boot")]]) for c in range(K) if n_b[c]]
    calib = [(c, X_all[class_idx[(c, "calib")]]) for c in range(K) if n_c[c]]
    dev = [(c, X_all[class_idx[(c, "dev")]]) for c in range(K)]
    fin = [(c, X_all[class_idx[(c, "final")]]) for c in range(K)] if include_final else None
    cent = {c: pools[c][:300] for c in range(K)}
    meta = dict(csv_classes=order, dedup_dropped=int(n_raw - int(keep0.sum())), entity_purged_rows=int(purged),
                split_type="csv_chronological" if t_all is not None else "csv_random_NOT_TEMPORAL",
                temporal_split=t_all is not None, entity_split=e_all is not None)
    return sample, boot, calib, dev, fin, cent, d, K, schedule, meta


def _cat(parts):
    X = np.vstack([x for _, x in parts])
    y = np.concatenate([np.full(len(x), c, dtype=np.int64) for c, x in parts])
    return X, y


def build_stream(cfg: DataCfg, seed: int, include_final: bool = False) -> StreamData:
    """include_final=False (development) never materialises the final test period."""
    rng = make_rng(seed, "data")
    meta: Dict[str, Any] = {"dataset": cfg.kind, "is_synthetic": cfg.kind == "synthetic"}
    K = cfg.n_classes
    if cfg.kind == "synthetic":
        sample = _synthetic_sampler(cfg, rng, seed)
        schedule = cfg.schedule
        boot = [(c, sample(c, cfg.bootstrap_per_class * (2 if c == 0 else 1), rng)) for c in cfg.base_classes]
        test = [(c, sample(c, cfg.test_per_class, rng)) for c in range(K)]
        cent_src = {c: sample(c, 300, rng) for c in range(1, K)}
        d = cfg.n_features
        rc = make_rng(seed, "calib_period")
        calib = [(c, sample(c, cfg.calib_per_class * (2 if c == 0 else 1), rc, "pre")) for c in cfg.base_classes]
        rp = make_rng(seed, "dev_post")
        test_post = [(c, sample(c, cfg.test_per_class, rp, "post")) for c in range(K)]
        fin = None
        if include_final:
            rf = make_rng(seed, "final_test")
            fin = [(c, sample(c, cfg.final_test_per_class, rf, "post")) for c in range(K)]
        meta.update(split_type="synthetic_iid_fresh_draws", temporal_split=False, dedup_dropped=0,
                    entity_split=False, entity_purged_rows=0)
        dev_pre_parts, dev_post_parts = test, test_post
    elif cfg.kind == "csv":
        sample, boot, calib, dev, fin, cent_src, d, K, schedule, m2 = _build_csv(cfg, rng, include_final)
        meta.update(m2)
        dev_pre_parts = dev_post_parts = dev
    else:
        raise ValueError(cfg.kind)
    chunks, next_uid = _build_chunks(cfg, sample, rng, schedule)
    _apply_label_noise(chunks, cfg.label_noise_rate, K, rng)
    bX, by = _cat(boot)
    cX, cy = _cat(calib)
    dev_pre_set = _cat(dev_pre_parts)
    dev_post_set = dev_pre_set if dev_post_parts is dev_pre_parts else _cat(dev_post_parts)
    eval_sets = {"dev_pre": dev_pre_set, "dev_post": dev_post_set}
    if fin is not None:
        eval_sets["final"] = _cat(fin)
    # ---- leakage guards: evaluation data must be disjoint from everything a learner can ever see
    stream_rows = np.vstack([c.X for c in chunks])
    trusted = np.vstack([bX, cX])
    for name, (eX, _) in eval_sets.items():
        assert_disjoint_rows(eX, trusted, f"eval set {name} and trusted bootstrap/calibration data")
        assert_disjoint_rows(eX, stream_rows, f"eval set {name} and stream chunks")
    assert_disjoint_rows(bX, cX, "bootstrap and calibration periods")
    assert_disjoint_rows(trusted, stream_rows, "bootstrap/calibration and stream chunks")
    names = list(eval_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if eval_sets[names[i]] is not eval_sets[names[j]]:
                assert_disjoint_rows(eval_sets[names[i]][0], eval_sets[names[j]][0], f"eval sets {names[i]}/{names[j]}")
    meta["overlap_check"] = "passed"
    benign_boot = bX[by == 0]
    mu, sd = benign_boot.mean(0), benign_boot.std(0) + 1e-9
    trig_idx = np.sort(rng.choice(d, size=min(cfg.trigger_features, d), replace=False))
    first, last = {}, {}
    for c, s_, e in schedule:
        first[c] = min(first.get(c, 10 ** 9), s_)
        last[c] = max(last.get(c, -1), e)
    meta.update(feature_mean=mu, feature_std=sd, next_uid=next_uid, schedule=tuple(schedule),
                class_first_chunk=first, class_last_chunk=last, base_classes=tuple(cfg.base_classes),
                label_noise_rate=cfg.label_noise_rate, drift_chunk=cfg.drift_chunk,
                has_final=fin is not None, calib_classes=sorted(set(int(c) for c in np.unique(cy))),
                eval_trigger=dict(idx=trig_idx.tolist(), vals=(mu[trig_idx] + cfg.trigger_sigma * sd[trig_idx]).tolist()),
                class_centroids={c: x.mean(0) for c, x in cent_src.items()})   # ATTACKER knowledge only
    return StreamData(bX, by, cX, cy, chunks, eval_sets, K, d, meta).freeze()
"""evcl.continual -- replay buffers, continual learner, kNN label sanitiser (B2 component)."""
from __future__ import annotations

import numpy as np

from .core import LearnerCfg, make_rng


class _Reservoir:
    def __init__(self, cap, d, rng):
        self.cap, self.rng, self.n, self.seen = cap, rng, 0, 0
        self.X = np.zeros((cap, d))
        self.y = np.zeros(cap, np.int64)
        self.i = np.zeros(cap, np.int64)

    def _put(self, i, x, y, oid):
        self.X[i], self.y[i], self.i[i] = x, y, oid

    def add(self, X, y, ids):
        m, i = len(y), 0
        room = self.cap - self.n
        if room > 0 and m:
            take = min(room, m)
            s = slice(self.n, self.n + take)
            self.X[s], self.y[s], self.i[s] = X[:take], y[:take], ids[:take]
            self.n += take
            self.seen += take
            i = take
        for j in range(i, m):
            self.seen += 1
            r = int(self.rng.integers(0, self.seen))
            if r < self.cap:
                self._put(r, X[j], y[j], ids[j])

    def view(self):
        return self.X[:self.n], self.y[:self.n], self.i[:self.n]


class ReplayBuffer:
    def __init__(self, capacity, n_features, n_classes, policy, rng):
        self.rng, self.policy, self.K = rng, policy, n_classes
        if policy == "reservoir":
            self.parts = {-1: _Reservoir(capacity, n_features, rng)}
        elif policy == "class_balanced":
            self.parts = {c: _Reservoir(max(1, capacity // n_classes), n_features, rng) for c in range(n_classes)}
        else:
            raise ValueError(policy)

    def add(self, X, y, ids):
        if self.policy == "reservoir":
            self.parts[-1].add(X, y, ids)
        else:
            for c in np.unique(y):
                m = y == c
                self.parts[int(c)].add(X[m], y[m], ids[m])

    def all(self):
        v = [r.view() for r in self.parts.values()]
        return tuple(np.concatenate([x[i] for x in v]) for i in range(3))

    def sample(self, n):
        X, y, _ = self.all()
        if len(y) == 0:
            return X, y
        idx = self.rng.choice(len(y), size=min(n, len(y)), replace=False)
        return X[idx], y[idx]

    def ids(self):
        return self.all()[2]


class ContinualLearner:
    """Linear (SGD, log-loss) class-incremental NIDS with experience replay. Output space is pre-allocated."""

    def __init__(self, cfg: LearnerCfg, n_classes: int, n_features: int, seed: int):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        self.cfg, self.K, self.d = cfg, n_classes, n_features
        self.rng = make_rng(seed, "learner")
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss="log_loss", alpha=cfg.alpha, learning_rate="constant", eta0=cfg.eta0,
                                   random_state=int(seed), shuffle=False)
        self.buffer = ReplayBuffer(cfg.replay_capacity, n_features, n_classes, cfg.buffer_policy,
                                   make_rng(seed, "buffer"))
        self.boot_X = self.boot_y = None
        self.n_updates = 0
        self._classes = np.arange(n_classes)

    def _fit(self, Xs, y, epochs):
        for _ in range(epochs):
            p = self.rng.permutation(len(y))
            self.model.partial_fit(Xs[p], y[p], classes=self._classes)

    def bootstrap(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self._fit(Xs, y, self.cfg.bootstrap_epochs)
        self.boot_X, self.boot_y = Xs, y.copy()
        self.buffer.add(Xs, y, np.full(len(y), -1, np.int64))   # -1 = trusted bootstrap

    def transform(self, X):
        return self.scaler.transform(X)

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))

    def predict(self, X):
        return self.model.predict(self.transform(X))

    def p_attack(self, X):
        return 1.0 - self.predict_proba(X)[:, 0]

    def loss_per_sample(self, X, y):
        P = self.predict_proba(X)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))

    def update(self, X, y, ids):
        if len(y) == 0:
            return
        self.n_updates += 1
        Xs = self.transform(X)
        rX, ry = self.buffer.sample(self.cfg.replay_batch)
        bx = np.vstack([Xs, rX]) if len(ry) else Xs
        by = np.concatenate([y, ry]) if len(ry) else y
        self._fit(bx, by, self.cfg.epochs)
        self.buffer.add(Xs, y, ids)


class SoftmaxLearner:
    """Linear softmax classifier (numpy minibatch SGD) with experience replay and a label space that GROWS from
    admitted labels: a class exists only after an admitted observation carries that label, so the learner never
    'knows' how many classes will appear. Labels outside [0, max_classes) are ignored (labels are attackable)."""

    def __init__(self, cfg: LearnerCfg, n_features: int, seed: int):
        self.cfg, self.d = cfg, n_features
        self.rng = make_rng(seed, "learner")
        self.classes: list = []
        self.cidx: dict = {}
        self.W = np.zeros((0, n_features))
        self.b = np.zeros(0)
        self.mu = self.sd = None
        self.buffer = ReplayBuffer(cfg.replay_capacity, n_features, cfg.max_classes, cfg.buffer_policy,
                                   make_rng(seed, "buffer"))
        self.boot_X = self.boot_y = None
        self.n_updates = 0
        self.n_invalid_labels = 0

    # -- label space
    def _grow(self, y):
        for c in np.unique(y):
            c = int(c)
            if c not in self.cidx:
                self.cidx[c] = len(self.classes)
                self.classes.append(c)
                self.W = np.vstack([self.W, np.zeros((1, self.d))])
                self.b = np.append(self.b, 0.0)

    def _valid(self, y):
        return (y >= 0) & (y < self.cfg.max_classes)

    # -- model
    def _softmax(self, Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def _fit(self, Xs, y, epochs):
        yi = np.array([self.cidx[int(c)] for c in y])
        bs, lr, l2 = self.cfg.batch_size, self.cfg.lr, self.cfg.l2
        w = None
        if self.cfg.class_balanced_updates:
            cnt = np.bincount(yi, minlength=len(self.classes)).astype(float)
            w = 1.0 / np.maximum(cnt[yi], 1.0)
            w = w / w.mean()
        for _ in range(epochs):
            perm = self.rng.permutation(len(yi))
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                P = self._softmax(Xs[j] @ self.W.T + self.b)
                P[np.arange(len(j)), yi[j]] -= 1.0
                if w is not None:
                    P *= w[j][:, None]
                self.W -= lr * (P.T @ Xs[j] / len(j) + l2 * self.W)
                self.b -= lr * P.mean(0)

    def bootstrap(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self._grow(y)
        self._fit(Xs, y, self.cfg.bootstrap_epochs)
        self.boot_X, self.boot_y = Xs, y.copy()
        self.buffer.add(Xs, y, np.full(len(y), -1, np.int64))

    def transform(self, X):
        return (X - self.mu) / self.sd

    def predict_proba(self, X):
        return self._softmax(self.transform(X) @ self.W.T + self.b)

    def predict(self, X):
        return np.asarray(self.classes)[self.predict_proba(X).argmax(1)]

    def p_attack(self, X):
        return 1.0 - self.predict_proba(X)[:, self.cidx[0]]

    def loss_per_sample(self, X, y):
        """-log p(y|x); NaN for labels the model does not know yet (undefined, not 'high loss')."""
        P = self.predict_proba(X)
        out = np.full(len(y), np.nan)
        known = np.array([int(c) in self.cidx for c in y])
        if known.any():
            idx = np.array([self.cidx[int(c)] for c in y[known]])
            out[known] = -np.log(np.clip(P[known][np.arange(known.sum()), idx], 1e-12, 1.0))
        return out

    def update(self, X, y, ids):
        if len(y) == 0:
            return
        ok = self._valid(y)
        self.n_invalid_labels += int((~ok).sum())
        X, y, ids = X[ok], y[ok], ids[ok]
        if len(y) == 0:
            return
        self.n_updates += 1
        Xs = self.transform(X)
        self._grow(y)
        rX, ry = self.buffer.sample(self.cfg.replay_batch)
        bx = np.vstack([Xs, rX]) if len(ry) else Xs
        by = np.concatenate([y, ry]) if len(ry) else y
        self._fit(bx, by, self.cfg.epochs)
        self.buffer.add(Xs, y, ids)


class MLPLearner(SoftmaxLearner):
    """Nonlinear learner: one hidden ReLU layer (HIDDEN units) + softmax output, minibatch SGD, the SAME replay buffer, label-space growth,
    class-balanced options and attackable-label bound as SoftmaxLearner. Used as a second model family (Phase 3 learner arm)."""
    HIDDEN = 32

    def __init__(self, cfg: LearnerCfg, n_features: int, seed: int):
        super().__init__(cfg, n_features, seed)
        H = self.HIDDEN
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / n_features), (H, n_features))
        self.b1 = np.zeros(H)
        self.W2 = np.zeros((0, H))
        self.b2 = np.zeros(0)

    def _grow(self, y):
        for c in np.unique(y):
            c = int(c)
            if c not in self.cidx:
                self.cidx[c] = len(self.classes)
                self.classes.append(c)
                self.W2 = np.vstack([self.W2, self.rng.normal(0.0, 0.1, (1, self.HIDDEN))])
                self.b2 = np.append(self.b2, 0.0)

    def _forward(self, Xs):
        Hh = np.maximum(Xs @ self.W1.T + self.b1, 0.0)
        return self._softmax(Hh @ self.W2.T + self.b2), Hh

    def predict_proba(self, X):
        return self._forward(self.transform(X))[0]

    def _fit(self, Xs, y, epochs):
        yi = np.array([self.cidx[int(c)] for c in y])
        bs, lr, l2 = self.cfg.batch_size, self.cfg.lr, self.cfg.l2
        w = None
        if self.cfg.class_balanced_updates:
            cnt = np.bincount(yi, minlength=len(self.classes)).astype(float)
            w = 1.0 / np.maximum(cnt[yi], 1.0)
            w = w / w.mean()
        for _ in range(epochs):
            perm = self.rng.permutation(len(yi))
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                P, Hh = self._forward(Xs[j])
                P[np.arange(len(j)), yi[j]] -= 1.0
                if w is not None:
                    P *= w[j][:, None]
                dH = (P @ self.W2) * (Hh > 0)
                self.W2 -= lr * (P.T @ Hh / len(j) + l2 * self.W2)
                self.b2 -= lr * P.mean(0)
                self.W1 -= lr * (dH.T @ Xs[j] / len(j) + l2 * self.W1)
                self.b1 -= lr * dH.mean(0)


class ForestLearner(SoftmaxLearner):
    """Tree-based learner: a random forest REFIT from scratch on the replay memory (bootstrap + admitted observations) after every update,
    the common 'periodic retraining' practice. Same buffer, label-space growth and attackable-label bound as SoftmaxLearner.
    Forest probabilities are coarse, so loss_per_sample is large for observations the forest votes against (documented limitation)."""
    N_TREES, MAX_DEPTH, MIN_LEAF = 100, 12, 2

    def __init__(self, cfg: LearnerCfg, n_features: int, seed: int):
        super().__init__(cfg, n_features, seed)
        self.seed = int(seed)
        self.model = None

    def _refit(self):
        from sklearn.ensemble import RandomForestClassifier
        X, y, _ = self.buffer.all()
        self.model = RandomForestClassifier(n_estimators=self.N_TREES, max_depth=self.MAX_DEPTH, min_samples_leaf=self.MIN_LEAF,
                                            class_weight="balanced_subsample" if self.cfg.class_balanced_updates else None,
                                            random_state=self.seed, n_jobs=2).fit(X, y)

    def bootstrap(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self._grow(y)
        self.boot_X, self.boot_y = Xs, y.copy()
        self.buffer.add(Xs, y, np.full(len(y), -1, np.int64))
        self._refit()

    def predict_proba(self, X):
        raw = self.model.predict_proba(self.transform(X))
        P = np.zeros((len(raw), len(self.classes)))
        for k, c in enumerate(self.model.classes_):
            P[:, self.cidx[int(c)]] = raw[:, k]
        return P

    def update(self, X, y, ids):
        if len(y) == 0:
            return
        ok = self._valid(y)
        self.n_invalid_labels += int((~ok).sum())
        X, y, ids = X[ok], y[ok], ids[ok]
        if len(y) == 0:
            return
        self.n_updates += 1
        Xs = self.transform(X)
        self._grow(y)
        self.buffer.add(Xs, y, ids)
        self._refit()


def make_learner(cfg, n_features: int, seed: int):
    if cfg.learner.algo == "softmax_dynamic":
        return SoftmaxLearner(cfg.learner, n_features, seed)
    if cfg.learner.algo == "mlp_dynamic":
        return MLPLearner(cfg.learner, n_features, seed)
    if cfg.learner.algo == "forest_refit":
        return ForestLearner(cfg.learner, n_features, seed)
    return ContinualLearner(cfg.learner, cfg.data.n_classes, n_features, seed)   # legacy: pre-allocated label space


class KNNSanitizer:
    """DEVELOPMENT PROXY, NOT a published-defence reproduction. kNN label sanitisation in the spirit of
    label-sanitisation defences (a possibly related paper: Paudice et al., 2018 -- citation NOT verified here). Keeps a point if >= min_agreement of its k nearest neighbours (trusted bootstrap +
    replay buffer + current chunk, excluding itself) share its label. NOT a faithful reproduction of any
    specific paper; replace with an established published defence in Phase 2 via the same interface."""

    def __init__(self, k=10, min_agreement=0.5):
        self.k, self.min_agreement = k, min_agreement

    def keep_mask(self, X, y, ref_X, ref_y):
        from sklearn.neighbors import NearestNeighbors
        n = len(y)
        if n == 0:
            return np.zeros(0, bool)
        allX, ally, off = np.vstack([ref_X, X]), np.concatenate([ref_y, y]), len(ref_y)
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(ally))).fit(allX)
        idx = nn.kneighbors(X, return_distance=False)
        keep = np.ones(n, bool)
        for i in range(n):
            row = idx[i][idx[i] != off + i][: self.k]
            if len(row):
                keep[i] = np.mean(ally[row] == y[i]) >= self.min_agreement
        return keep
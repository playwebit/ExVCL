"""exvcl.api -- the gate plus a bundled continual learner, for the common case where you do not bring your own model."""
from __future__ import annotations

from typing import Optional

import numpy as np

from evcl.continual import ForestLearner, MLPLearner, SoftmaxLearner
from evcl.core import LearnerCfg

from .evidence import EvidenceModel
from .gate import AdmissionGate, GateResult, VerifyFn

_ALGOS = {"softmax": ("softmax_dynamic", SoftmaxLearner), "mlp": ("mlp_dynamic", MLPLearner), "forest": ("forest_refit", ForestLearner)}


def make_learner(kind: str, n_features: int, seed: int = 0, **params):
    """Class-incremental learner with replay memory: 'softmax' (the learner of the paper), 'mlp' or 'forest'.
    params override LearnerCfg fields (lr, batch_size, l2, epochs, bootstrap_epochs, replay_capacity, replay_batch, max_classes, ...)."""
    if kind not in _ALGOS:
        raise ValueError(f"learner must be one of {tuple(_ALGOS)}")
    algo, cls = _ALGOS[kind]
    return cls(LearnerCfg(algo=algo, **params), n_features, seed)


class ExVCL:
    """ExVCL = admission gate + continual learner. Typical use:

        ev = EvidenceModel().fit(F_cal, y_cal_malicious)
        ex = ExVCL(n_features=20, mode="strict", evidence=ev)
        ex.bootstrap(X0, y0)                                   # trusted initial data
        for X, y_obs, F in stream:                             # F: evidence counts of each observation (or a callable)
            res = ex.process_chunk(X, y_obs, verify=lambda idx: (F[idx], np.ones(len(idx), bool)))
    """

    def __init__(self, n_features: int, mode: str = "strict", evidence: Optional[EvidenceModel] = None, learner: str = "softmax",
                 budget=0.3, audit_rate: float = 0.05, triage_low: float = 0.2, seed: int = 0, learner_params: Optional[dict] = None,
                 **gate_kwargs):
        self.n_features = int(n_features)
        self.gate = AdmissionGate(mode, evidence, budget, audit_rate, triage_low, seed=seed, **gate_kwargs)
        self.learner = make_learner(learner, self.n_features, seed, **(learner_params or {}))
        self._next_id = 0
        self._boot = False

    @property
    def mode(self) -> str:
        return self.gate.mode

    def bootstrap(self, X, y) -> None:
        """Initial fit on TRUSTED labelled data (no gate)."""
        self.learner.bootstrap(np.asarray(X, float), np.asarray(y).astype(int))
        self._boot = True

    def calibrate_triage(self, X_benign_cal) -> float:
        """Selective mode: fast-path threshold from the model's attack scores on benign calibration data."""
        return self.gate.calibrate_triage(self.learner.p_attack(np.asarray(X_benign_cal, float)))

    def process_chunk(self, X, y_obs, verify: Optional[VerifyFn] = None) -> GateResult:
        if not self._boot:
            raise RuntimeError("call bootstrap(X, y) with trusted data first")
        X, y_obs = np.asarray(X, float), np.asarray(y_obs).astype(int)
        if X.ndim != 2 or X.shape[1] != self.n_features or len(X) != len(y_obs):
            raise ValueError(f"X must have shape (n, {self.n_features}) and match y_obs")
        p = self.learner.p_attack(X) if self.gate.mode == "selective" else None
        res = self.gate.decide(y_obs, verify, p)
        ids = np.arange(self._next_id, self._next_id + len(y_obs))
        self._next_id += len(y_obs)
        self.learner.update(X[res.admit], res.label[res.admit], ids[res.admit])
        return res

    def predict(self, X):
        return self.learner.predict(np.asarray(X, float))

    def predict_proba(self, X):
        return self.learner.predict_proba(np.asarray(X, float))

    def p_attack(self, X):
        return self.learner.p_attack(np.asarray(X, float))

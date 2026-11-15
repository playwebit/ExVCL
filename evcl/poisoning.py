"""evcl.poisoning -- attack laboratory.

All attacks are simple, documented heuristics operating on the labelled observation stream. They are
NOT claimed to be optimal or state-of-the-art; Phase 3 should add published attacks (e.g. optimisation-
based clean-label / backdoor attacks on continual NIDS) behind the same interface.
Attacks are applied ONCE per (seed, attack) so every pipeline sees an identical attacked stream."""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from .data import EVASION_MODES, KIND, Chunk, StreamData


def _clone(stream: StreamData) -> StreamData:
    return StreamData(stream.bootstrap_X, stream.bootstrap_y, stream.calib_X, stream.calib_y,
                      [c.copy() for c in stream.chunks], stream.eval_sets, stream.n_classes, stream.n_features,
                      copy.deepcopy(stream.meta))


def _inject(out: StreamData, t: int, X, y_obs, y_true, kind: str, rng, evasive=None, trigger=None, evasion=None):
    n = len(X)
    if n == 0:
        return
    u0 = out.meta["next_uid"]
    out.meta["next_uid"] = u0 + n
    new = Chunk(X=X, y_obs=y_obs.astype(np.int64), y_true=y_true.astype(np.int64), is_poison=np.ones(n, bool),
                kind=np.full(n, KIND[kind], np.int8),
                evasive=np.zeros(n, bool) if evasive is None else evasive,
                trigger=np.zeros(n, bool) if trigger is None else trigger,
                uid=np.arange(u0, u0 + n, dtype=np.int64),
                evasion=np.zeros(n, np.int8) if evasion is None else evasion)
    merged = out.chunks[t].append(new)
    out.chunks[t] = merged.take(rng.permutation(len(merged)))


def _attack_copies(out: StreamData, ch: Chunk, n: int, rng, jitter=0.05):
    cand = np.flatnonzero((ch.y_true > 0) & (~ch.is_poison))
    if len(cand) == 0 or n == 0:
        return None, None
    idx = rng.choice(cand, size=n, replace=True)
    X = ch.X[idx] + rng.normal(size=(n, ch.X.shape[1])) * jitter * out.meta["feature_std"]
    return X, ch.y_true[idx]


class Attack:
    name = "none"

    def __init__(self, **params):
        self.params = params

    def apply(self, stream: StreamData, rng) -> StreamData:
        return _clone(stream)


def _window(p, n_chunks):
    return int(p.get("start", 0)), min(int(p.get("end", n_chunks)), n_chunks)


class LabelPoisoning(Attack):
    """Relabel a fraction of true attack observations as benign (targeted attack->benign flips)."""
    name, _kind = "label_poisoning", "label"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        evf = float(self.params.get("evasive_fraction", 0.0))
        for t in range(s, e):
            ch = out.chunks[t]
            cand = np.flatnonzero((ch.y_true > 0) & (~ch.is_poison))
            n = min(len(cand), int(round(self.params["budget"] * len(ch))))
            if n == 0:
                continue
            idx = rng.choice(cand, size=n, replace=False)
            ch.y_obs[idx] = 0
            ch.is_poison[idx] = True
            ch.kind[idx] = KIND[self._kind]
            if evf > 0:
                ev = rng.random(n) < evf
                ch.evasive[idx] = ev
                mode = self.params.get("evasion_mode", "mixed")
                codes = (rng.integers(1, 5, size=n) if mode == "mixed" else np.full(n, EVASION_MODES[mode]))
                ch.evasion[idx] = np.where(ev, codes, 0).astype(np.int8)
        return out


class SlowPoisoning(LabelPoisoning):
    """Same flip mechanism at a very low per-chunk rate spread over the whole stream."""
    name, _kind = "slow_poisoning", "slow"


class SandboxEvasionPoisoning(LabelPoisoning):
    """Label flips on samples whose malicious behaviour tries to evade the sandbox."""
    name, _kind = "sandbox_evasion", "evasion"


class CleanLabelPoisoning(Attack):
    """Inject correctly-labelled benign-looking points shifted toward a target attack family's centroid
    (white-box attacker knowledge of the centroid). Their TRUE behaviour is benign."""
    name = "clean_label"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        cen = out.meta["class_centroids"][int(self.params["target_class"])]
        a = float(self.params["alpha"])
        for t in range(s, e):
            ch = out.chunks[t]
            cand = np.flatnonzero((ch.y_true == 0) & (~ch.is_poison))
            n = int(round(self.params["budget"] * len(ch)))
            if len(cand) == 0 or n == 0:
                continue
            idx = rng.choice(cand, size=n, replace=True)
            X = ch.X[idx] + a * (cen - ch.X[idx])
            z = np.zeros(n, np.int64)
            _inject(out, t, X, z, z, "clean_label", rng)
        return out


class ReplayFlood(Attack):
    """Burst of attack-like observations labelled benign to dominate the replay memory."""
    name = "replay_flood"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        for t in range(s, e):
            ch = out.chunks[t]
            n = int(round(self.params["flood_factor"] * len(ch)))
            X, yt = _attack_copies(out, ch, n, rng)
            if X is not None:
                _inject(out, t, X, np.zeros(len(X), np.int64), yt, "replay_flood", rng)
        return out


class TriageAwarePoisoning(Attack):
    """ADAPTIVE (white-box) proxy: label-flips the attack observations that look MOST benign (closest to the benign centroid
    relative to their own family's centroid), because those are the ones a confidence-based fast path will not send to
    verification. Attacker knowledge: benign mean and family centroids (same white-box assumption as clean_label).
    It does not query the defender model; it is a feature-space proxy for 'target the fast path'."""
    name = "triage_aware_label"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        mu = out.meta["feature_mean"]
        cents = out.meta["class_centroids"]
        for t in range(s, e):
            ch = out.chunks[t]
            cand = np.flatnonzero((ch.y_true > 0) & (~ch.is_poison))
            n = min(len(cand), int(round(self.params["budget"] * len(ch))))
            if n == 0:
                continue
            db = np.linalg.norm(ch.X[cand] - mu, axis=1)
            dc = np.array([np.linalg.norm(ch.X[i] - cents[int(ch.y_true[i])]) for i in cand])
            order = np.argsort(db / (db + dc + 1e-12), kind="stable")      # most benign-looking first
            idx = cand[order[:n]]
            ch.y_obs[idx] = 0
            ch.is_poison[idx] = True
            ch.kind[idx] = KIND["triage_aware"]
        return out


class ReplayDuplicate(Attack):
    """EXACT duplicates of a few captured attack observations, replayed many times with a benign label.
    Distinct from ReplayFlood (jittered copies): exact duplicates are detectable by de-duplication, which no
    pipeline here uses (so this attack measures duplicate-driven buffer/gradient domination)."""
    name = "replay_duplicate"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        rep = int(self.params["repeats"])
        for t in range(s, e):
            ch = out.chunks[t]
            cand = np.flatnonzero((ch.y_true > 0) & (~ch.is_poison))
            n_distinct = max(1, int(round(self.params["budget"] * len(ch) / rep)))
            if len(cand) == 0:
                continue
            idx = np.repeat(rng.choice(cand, size=min(n_distinct, len(cand)), replace=False), rep)
            _inject(out, t, ch.X[idx].copy(), np.zeros(len(idx), np.int64), ch.y_true[idx], "replay_duplicate", rng)
        return out


class Backdoor(Attack):
    """Attack observations carrying a fixed feature trigger, labelled benign."""
    name = "backdoor"

    def apply(self, stream, rng):
        out = _clone(stream)
        s, e = _window(self.params, len(out.chunks))
        trig = out.meta["eval_trigger"]
        idx_f, vals = np.asarray(trig["idx"]), np.asarray(trig["vals"])
        for t in range(s, e):
            ch = out.chunks[t]
            X, yt = _attack_copies(out, ch, int(round(self.params["budget"] * len(ch))), rng)
            if X is None:
                continue
            X[:, idx_f] = vals
            _inject(out, t, X, np.zeros(len(X), np.int64), yt, "backdoor", rng,
                    trigger=np.ones(len(X), bool))
        return out


ATTACKS = {c.name: c for c in (LabelPoisoning, SlowPoisoning, SandboxEvasionPoisoning,
                               CleanLabelPoisoning, ReplayFlood, ReplayDuplicate, Backdoor, TriageAwarePoisoning)}


ATTACK_SPEC = {
    "label_poisoning": dict(
        threat_model="attacker controls the labelling channel for a fraction of observations", capability="flip labels only; features unchanged",
        rate="budget x chunk size per chunk inside the window", target="true attack observations labelled benign",
        assumptions="attacker knows which observations are attacks; labels arrive with observations",
        success="macro-F1 loss on the clean test period (PI); rise of FNR", behavioral_detectable="yes (true behaviour is malicious)"),
    "clean_label": dict(
        threat_model="attacker injects correctly labelled benign traffic", capability="craft features (white-box centroid knowledge)",
        rate="budget x chunk size injected per chunk", target="shift the benign region toward a target attack family",
        assumptions="crafted traffic is TRULY benign in behaviour", success="PI / rise of FNR on the target family",
        behavioral_detectable="NO by construction (behaviourally indistinguishable from legitimate traffic)"),
    "replay_flood": dict(
        threat_model="attacker floods the learner with mislabelled near-copies", capability="inject jittered copies of attack traffic labelled benign",
        rate="flood_factor x chunk size inside a short window", target="dominate replay memory / gradients", assumptions="no de-duplication",
        success="PI; buffer contamination", behavioral_detectable="yes (copies of malicious behaviour)"),
    "replay_duplicate": dict(
        threat_model="attacker replays captured events", capability="repeat EXACT copies of a few attack observations with a benign label",
        rate="budget x chunk size per chunk (budget/repeats distinct events x repeats)", target="duplicate-driven domination of memory",
        assumptions="no de-duplication", success="PI; buffer contamination", behavioral_detectable="yes (copies of malicious behaviour)"),
    "backdoor": dict(
        threat_model="attacker plants a feature trigger with benign labels", capability="inject attack traffic with a fixed trigger pattern, labelled benign",
        rate="budget x chunk size per chunk", target="triggered malicious traffic classified benign",
        assumptions="fixed, hand-chosen trigger (not optimised)", success="ASR_cond and delta vs clean run (see evaluation.py)",
        behavioral_detectable="yes (underlying behaviour malicious)"),
    "slow_poisoning": dict(
        threat_model="low-rate label flipping over the whole stream", capability="as label_poisoning",
        rate="1% of chunk size per chunk", target="gradual PI while staying under naive rate thresholds",
        assumptions="no adaptive evasion of anomaly thresholds is modelled", success="PI", behavioral_detectable="yes"),
    "triage_aware_label": dict(
        threat_model="adaptive white-box attacker who knows the gate routes confident-normal traffic past verification",
        capability="label-flip only the attack observations that look most benign (feature-space proxy)",
        rate="budget x chunk size per chunk", target="poison that takes the unverified fast path",
        assumptions="knows benign mean and family centroids; does not query the defender model", success="PI / PAR on the fast path",
        behavioral_detectable="yes if verified; the point is to avoid being verified"),
    "sandbox_evasion": dict(
        threat_model="malware evades the verification sandbox", capability="label flips + latent evasion strategy (env-sensitive, delayed, mimicry, partial)",
        rate="budget x chunk size per chunk", target="get mislabelled malware admitted",
        assumptions="SIMULATION ONLY: modelled by simulator latents, NOT demonstrated against a real sandbox",
        success="VER and PI", behavioral_detectable="depends on telemetry fidelity and evasion strategy"),
}

ATTACK_STATUS = {
    "label_poisoning": "heuristic proxy: attack->benign label flips (not a published attack)",
    "clean_label": "heuristic proxy: benign points shifted toward a centroid (white-box); NOT feature-collision/optimisation based",
    "replay_flood": "heuristic proxy: burst of mislabelled copies (flooding only; no strategic buffer placement)",
    "backdoor": "heuristic proxy: fixed feature trigger + benign label; NOT optimised",
    "slow_poisoning": "heuristic proxy: low constant rate; does NOT model adaptive evasion of anomaly thresholds",
    "replay_duplicate": "heuristic proxy: exact duplicates of captured events with a benign label",
    "triage_aware_label": "heuristic adaptive proxy: targets the most benign-looking attack observations (feature-space, white-box centroids)",
    "sandbox_evasion": "SIMULATION-ONLY: needs SimulatedSandbox 'evasive' latent; no real evasion technique implemented",
}


def make_attack(name: str, params: Dict) -> Attack:
    if name == "none":
        return Attack()
    return ATTACKS[name](**params)
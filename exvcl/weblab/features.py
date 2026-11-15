"""Numeric features of a request for the continual learner (the model ExVCL protects). Computed from the raw request text."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .payloads import MARKER

FEATURE_NAMES = ("length", "dotdot", "slashes", "percent", "pct25", "dots", "digit_ratio", "upper_ratio", "depth", "sensitive_words",
                 "special_chars", "entropy", "overlong", "marker")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def learner_features(name: str, ref: str = "") -> np.ndarray:
    s = name
    low = s.lower()
    n = max(len(s), 1)
    return np.array([
        len(s), s.count(".."), s.count("/"), s.count("%"), low.count("%25"), s.count("."),
        sum(ch.isdigit() for ch in s) / n, sum(ch.isupper() for ch in s) / n, s.strip("/").count("/"),
        sum(low.count(w) for w in ("passwd", "shadow", "id_rsa", "sudoers", "authorized_keys", "/etc/", "etc%2f", "etc%252f", ".ssh")),
        sum(not (ch.isalnum() or ch in "._-/") for ch in s), _entropy(s), low.count("%c0"), float(MARKER in ref),
    ], float)


def matrix(events) -> np.ndarray:
    return np.vstack([learner_features(e.name, e.ref) for e in events])

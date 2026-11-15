"""The independent upstream classifier: character n-gram TF-IDF + logistic regression on the request text. It is NOT part of ExVCL; its
predictions are the 'observed labels' that the admission gate checks against execution evidence."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def text_of(e) -> str:
    return f"{e.name} | {e.ref}"


class UpstreamModel:
    def __init__(self, seed: int = 0, ngram=(2, 4), C: float = 3.0):
        self.pipe = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=ngram, min_df=2, sublinear_tf=True),
                                  LogisticRegression(C=C, max_iter=2000, random_state=seed))

    def fit(self, events) -> "UpstreamModel":
        self.pipe.fit([text_of(e) for e in events], np.array([e.ground_truth for e in events]))
        return self

    def predict(self, events) -> np.ndarray:
        return self.pipe.predict([text_of(e) for e in events]).astype(int)

    def predict_proba(self, events) -> np.ndarray:
        return self.pipe.predict_proba([text_of(e) for e in events])[:, 1]

    def evaluate(self, events) -> Dict:
        y = np.array([e.ground_truth for e in events]); p = self.predict(events)
        fam: Dict[str, float] = {}
        for f in sorted({e.family for e in events}):
            m = np.array([e.family == f for e in events]); fam[f] = float((p[m] == y[m]).mean())
        return dict(accuracy=float((p == y).mean()), attack_recall=float(p[y == 1].mean()), benign_false_positive=float(p[y == 0].mean()),
                    by_family=fam)

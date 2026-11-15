"""exvcl.weblab.serve -- live demo: every POSTed request is labelled by the upstream text model, executed in the strace sandbox, and passed
through three ExVCL instances (none / selective / strict) that each update their own continual learner. Binds to 127.0.0.1 by default.

    POST /request   {"name": "../../etc/passwd", "ref": "home", "label": 0 (optional: overrides the upstream label = poisoned labelling channel),
                     "ground_truth": 1 (optional, only for the statistics)}
    GET  /stats     counts of admitted / rejected / deferred per mode
    GET  /health

The request is executed only in the sandbox worker (it opens a path and reports found/size); the web process never touches the requested path.
Live mode decides one request at a time, so every request is verified in strict mode (budget 100 %); budget sweeps are in `exvcl weblab run`.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict

import numpy as np

from evcl.admission import REASON_NAMES
from ..api import ExVCL
from ..evidence import EvidenceModel
from ..telemetry import EVENT_KINDS
from . import payloads as P
from . import sandbox, target
from .features import learner_features, matrix
from .upstream import UpstreamModel

MODES = ("none", "selective", "strict")


def create_app(out_dir: str, seed: int = 0, boot_n: int = 60, cal_n: int = 80, workers: int = 2, learner: str = "softmax"):
    from flask import Flask, jsonify, request
    os.makedirs(out_dir, exist_ok=True)
    target.ensure_lab()
    upstream = UpstreamModel(seed).fit(P.generate(600, seed + 11, attack_families=("A1", "A2", "A3")))
    boot = P.generate(boot_n, seed + 21)
    cal = P.generate(cal_n, seed + 22, attack_families=P.EFFECTIVE_EVIDENCE)
    F_cal, ok_cal, _ = sandbox.collect_evidence([e.name for e in cal], workers)
    y_cal = np.array([e.ground_truth for e in cal])
    ev = EvidenceModel(seed=seed, drop=("syscall_total",)).fit(F_cal[ok_cal], y_cal[ok_cal])
    Xb, yb = matrix(boot), np.array([e.ground_truth for e in boot])
    systems = {}
    for m in MODES:
        ex = ExVCL(Xb.shape[1], m, ev if m != "none" else None, learner, budget=1.0, seed=seed)
        ex.bootstrap(Xb, yb)
        if m == "selective":
            ex.calibrate_triage(Xb[yb == 0])
        systems[m] = ex
    lock = threading.Lock(); stats = defaultdict(Counter); log_path = os.path.join(out_dir, "live_requests.jsonl"); wd = tempfile.mkdtemp(prefix="exvcl_live_")
    app = Flask("exvcl_weblab"); counter = [0]

    @app.get("/health")
    def health():
        return jsonify(ok=True, upstream_ready=True, evidence=ev.describe())

    @app.get("/stats")
    def get_stats():
        with lock:
            return jsonify(modes={m: dict(c) for m, c in stats.items()}, requests=counter[0])

    @app.post("/request")
    def handle():
        body = request.get_json(force=True, silent=True) or {}
        name, ref = str(body.get("name", "")), str(body.get("ref", ""))
        if not name or len(name) > 2000:
            return jsonify(error="name is required (max 2000 characters)"), 400
        evt = P.Event("", name, ref, int(body.get("ground_truth", -1)), "live")
        with lock:
            counter[0] += 1; rid = f"R{counter[0]:06d}"
            up = int(upstream.predict([evt])[0]); label = int(body["label"]) if "label" in body else up
            F, ok, secs = sandbox.run_sandbox(name, wd, rid)
            x = learner_features(name, ref)[None, :]
            verify = lambda idx: (F[None, :], np.array([ok]))                          # noqa: E731
            out = {}
            for m, ex in systems.items():
                r = ex.process_chunk(x, np.array([label]), None if m == "none" else verify)
                out[m] = dict(admitted=bool(r.admit[0]), reason=REASON_NAMES[int(r.reason[0])], verified=bool(r.n_verified), verdict=int(r.verdict[0]))
                stats[m]["requests"] += 1; stats[m]["admitted"] += int(r.admit[0]); stats[m]["verified"] += int(r.n_verified)
                stats[m][REASON_NAMES[int(r.reason[0])]] += 1
            rec = dict(id=rid, t=time.time(), name=name, ref=ref, upstream_label=up, observed_label=label, ground_truth=evt.ground_truth,
                       evidence=dict(zip(EVENT_KINDS, F.astype(int).tolist())), sandbox_ok=bool(ok), sandbox_s=round(secs, 3), modes=out)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return jsonify(rec)

    app.config["EXVCL_SYSTEMS"] = systems
    return app

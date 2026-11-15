"""exvcl command line.

    exvcl version
    exvcl selftest [--no-engine]
    exvcl demo [--seed N] [--sessions N] [--json FILE]
    exvcl vm-check [--sessions N] [--seed N] [--workdir DIR] [--keep] [--json FILE]
    exvcl eval-recording --sessions sessions.csv (--archive RUN.tar.gz | --run-dir FOLDER) [--sources strace,falco,auditd]
    exvcl weblab run --out DIR [--events 400 --seeds 20 --covered-only] | serve --out DIR | sample
    exvcl parse-strace FILE
"""
from __future__ import annotations

import argparse
import platform
import sys


def _version() -> int:
    import numpy, sklearn
    import evcl
    from evcl.core import source_hash
    import os
    from . import __version__
    print(f"exvcl {__version__}")
    print(f"engine evcl {evcl.__version__} (source hash {source_hash(os.path.dirname(os.path.abspath(evcl.__file__)))})")
    print(f"python {platform.python_version()}  numpy {numpy.__version__}  scikit-learn {sklearn.__version__}")
    return 0


def _selftest(engine: bool) -> int:
    from .selftest import run_all
    res = run_all(engine)
    bad = {k: v for k, v in res.items() if v != "PASS"}
    for k, v in res.items():
        print(("PASS  " if v == "PASS" else "FAIL  ") + k + ("" if v == "PASS" else f"   {v}"))
    print(f"\n{len(res) - len(bad)}/{len(res)} tests passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exvcl", description="ExVCL: execution-verified admission control for continual NIDS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print versions and the engine source hash")
    s = sub.add_parser("selftest", help="run the library tests (and the research engine's tests)")
    s.add_argument("--no-engine", action="store_true", help="skip the bundled research engine's own tests")
    d = sub.add_parser("demo", help="end-to-end demo on synthetic evidence")
    d.add_argument("--seed", type=int, default=0); d.add_argument("--sessions", type=int, default=80, help="sessions per class")
    d.add_argument("--repeats", type=int, default=5); d.add_argument("--json", default=None)
    v = sub.add_parser("vm-check", help="end-to-end check on REAL strace telemetry recorded on this machine (needs strace)")
    v.add_argument("--sessions", type=int, default=40, help="sessions per class (at least 20)")
    v.add_argument("--seed", type=int, default=0); v.add_argument("--repeats", type=int, default=5)
    v.add_argument("--workdir", default=None); v.add_argument("--keep", action="store_true")
    v.add_argument("--json", default="exvcl_vm_report.json")
    r = sub.add_parser("eval-recording", help="run the library on a real lab recording (run folder or .tar.gz + sessions.csv)")
    r.add_argument("--sessions", required=True); r.add_argument("--archive"); r.add_argument("--run-dir")
    r.add_argument("--sources", default="strace,falco,auditd"); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--repeats", type=int, default=5); r.add_argument("--include-hard", action="store_true")
    r.add_argument("--json", default="exvcl_recording_report.json")
    w = sub.add_parser("weblab", help="web-attack lab: run the poisoning experiment, serve a live demo, print sample requests")
    ws = w.add_subparsers(dest="wcmd", required=True)
    wr = ws.add_parser("run", help="experiment with budget sweep and evidence controls (needs strace)")
    wr.add_argument("--out", required=True); wr.add_argument("--events", type=int, default=400); wr.add_argument("--seeds", type=int, default=20)
    wr.add_argument("--budgets", default="0.3,0.5,0.7,1.0"); wr.add_argument("--label-flip-rate", type=float, default=0.2)
    wr.add_argument("--backdoor-rate", type=float, default=0.05); wr.add_argument("--workers", type=int, default=0)
    wr.add_argument("--chunk", type=int, default=20); wr.add_argument("--learner", default="softmax", choices=["softmax", "mlp", "forest"])
    wr.add_argument("--covered-only", action="store_true", help="stream only has request families whose evidence is informative (no blind spots)")
    wv = ws.add_parser("serve", help="live demo app (needs flask and strace)")
    wv.add_argument("--out", required=True); wv.add_argument("--port", type=int, default=5050); wv.add_argument("--host", default="127.0.0.1")
    wp = ws.add_parser("sample", help="print sample requests as JSON lines")
    wp.add_argument("--n", type=int, default=6); wp.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("parse-strace", help="print the seven evidence counts of a strace -f log")
    p.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "version":
        return _version()
    if a.cmd == "selftest":
        return _selftest(not a.no_engine)
    if a.cmd == "demo":
        from .checks import run_demo
        return 0 if run_demo(a.seed, a.sessions, a.repeats, a.json) else 1
    if a.cmd == "vm-check":
        from .checks import run_vm_check
        return run_vm_check(a.sessions, a.seed, a.workdir, a.keep, a.json, a.repeats)
    if a.cmd == "eval-recording":
        from .evalrec import run_eval_recording
        return run_eval_recording(a.sessions, a.archive, a.run_dir, tuple(a.sources.split(",")), a.seed, a.repeats, a.include_hard, json_path=a.json)
    if a.cmd == "weblab":
        import json, os
        if a.wcmd == "sample":
            from .weblab import payloads as P
            for e in P.generate(a.n, a.seed):
                print(json.dumps(dict(name=e.name, ref=e.ref, ground_truth=e.ground_truth, family=e.family)))
            return 0
        if a.wcmd == "run":
            from .weblab.experiment import run_experiment
            cov = dict(stream_attack_families=("A1", "A2", "A3", "A4", "A5"), stream_benign_families=("B1", "B2", "B3", "B4")) if a.covered_only else {}
            run_experiment(a.out, a.events, a.seeds, tuple(float(x) for x in a.budgets.split(",")), a.label_flip_rate, a.backdoor_rate, a.chunk,
                           a.workers or (os.cpu_count() or 2), learner=a.learner, **cov)
            return 0
        if a.wcmd == "serve":
            from .weblab.serve import create_app
            app = create_app(a.out)
            print(f"ExVCL web lab listening on http://{a.host}:{a.port}  (POST /request, GET /stats); log: {a.out}/live_requests.jsonl")
            app.run(host=a.host, port=a.port, threaded=True)
            return 0
    if a.cmd == "parse-strace":
        from .telemetry import EVENT_KINDS, counts_from_strace_file
        c = counts_from_strace_file(a.file)
        for k, x in zip(EVENT_KINDS, c):
            print(f"{k:24s}{int(x)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

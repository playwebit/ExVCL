"""evcl.phase3_selftest -- tests of the Phase 3 collectors' parsers (strace / Falco / auditd) on synthetic logs rendered from ONE event script.
The renderers below are TEST FIXTURES that imitate each tool's output format; they are not recordings. The real formats are checked by the
on-VM smoke test (verify_collectors.sh)."""
from __future__ import annotations

import json
import time

import numpy as np

T0 = 1_790_000_000.0
MASTER, VICTIM_UID = 100, 1001
# (dt, pid, ppid, user, uid, exe, kind, arg)  kind: exec | open | connect | priv
SCRIPT = [
    (0.10, 200, 100, "victim", VICTIM_UID, "/usr/bin/bash", "exec", None),
    (0.11, 200, 100, "victim", VICTIM_UID, "/usr/bin/bash", "open", ("/etc/passwd", "O_RDONLY|O_CLOEXEC", 0x0)),
    (0.12, 200, 100, "victim", VICTIM_UID, "/usr/bin/bash", "open", ("/home/victim/out.txt", "O_WRONLY|O_CREAT|O_TRUNC", 0x241)),
    (0.13, 201, 200, "victim", VICTIM_UID, "/usr/bin/ls", "exec", None),
    (0.14, 201, 200, "victim", VICTIM_UID, "/usr/bin/ls", "open", ("/var/www/html/index.html", "O_RDONLY", 0x0)),
    (0.15, 150, 100, "root", 0, "/usr/sbin/sshd", "connect", ("10.10.0.3", 53)),
    (0.16, 150, 100, "root", 0, "/usr/sbin/sshd", "priv", ("setuid", 1001)),
    (0.17, 150, 100, "root", 0, "/usr/sbin/sshd", "open", ("/etc/shadow", "O_RDONLY", 0x0)),
]
MGMT = (0.20, 900, 1, "admin", 1000, "/usr/sbin/sshd", "open", ("/etc/shadow", "O_RDONLY", 0x0))      # management session: out of scope
_SYSNO = {"exec": 59, "open": 257, "connect": 42, "setuid": 105}
KINDS = ("process_spawn", "shell_exec", "file_write", "sensitive_file_access", "net_connect_out", "priv_change", "syscall_total")


def render_strace(script):
    out = []
    for dt, pid, ppid, user, uid, exe, kind, arg in script:
        t = f"{pid}  {T0 + dt:.6f}"
        if kind == "exec":
            out.append(f'{t} execve("{exe}", ["{exe}"], 0x7ffd /* 20 vars */) = 0')
        elif kind == "open":
            out.append(f'{t} openat(AT_FDCWD, "{arg[0]}", {arg[1]}) = 3')
        elif kind == "connect":
            out.append(f'{t} connect(3, {{sa_family=AF_INET, sin_port=htons({arg[1]}), sin_addr=inet_addr("{arg[0]}")}}, 16) = 0')
        elif kind == "priv":
            out.append(f"{t} {arg[0]}({arg[1]}) = 0")
    return "\n".join(out)


def render_falco(script):
    out = []
    for dt, pid, ppid, user, uid, exe, kind, arg in script:
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(T0 + dt)) + f".{int(round(((T0 + dt) % 1) * 1e9)):09d}Z"
        f = {"proc.pid": pid, "proc.ppid": ppid, "proc.name": exe.split("/")[-1], "proc.exepath": exe, "user.name": user}
        if kind == "exec":
            f["evt.type"] = "execve"
        elif kind == "open":
            f.update({"evt.type": "openat", "fd.name": arg[0], "evt.arg.flags": arg[1]})
        elif kind == "connect":
            f.update({"evt.type": "connect", "fd.type": "ipv4", "fd.name": f"10.10.0.2:40000->{arg[0]}:{arg[1]}"})
        elif kind == "priv":
            f["evt.type"] = arg[0]
        out.append(json.dumps({"time": iso, "rule": "evcl_events", "priority": "Debug", "output": "x", "output_fields": f}))
    return "\n".join(out)


def render_audit(script):
    out, ser = [], 1000
    for dt, pid, ppid, user, uid, exe, kind, arg in script:
        ser += 1
        mid = f"audit({T0 + dt:.3f}:{ser})"
        name = arg[0] if kind == "priv" else kind
        sysno = 105 if kind == "priv" else _SYSNO[kind]
        a1 = f"{arg[2]:x}" if kind == "open" else "0"
        out.append(f'type=SYSCALL msg={mid}: arch=c000003e syscall={sysno} success=yes exit=3 a0=ffffff9c a1=0 a2={a1} items=1 ppid={ppid} pid={pid} auid=1000 uid={uid} comm="{exe.split("/")[-1]}" exe="{exe}" key="evcl"')
        if kind == "open":
            out.append(f'type=PATH msg={mid}: item=0 name="{arg[0]}" inode=1 dev=08:01')
        elif kind == "exec":
            out.append(f'type=PATH msg={mid}: item=0 name="{exe}" inode=1 dev=08:01')
        elif kind == "connect":
            ip = "".join(f"{int(x):02X}" for x in arg[0].split("."))
            out.append(f"type=SOCKADDR msg={mid}: saddr=0200{arg[1]:04X}{ip}0000000000000000")
        out.append(f"type=PROCTITLE msg={mid}: proctitle=00")
    return "\n".join(out)


def _sums(ts_M):
    return np.asarray(ts_M[1]).sum(0).astype(int)


def run_all():
    import falco_parse as FP
    import audit_parse as AP
    import strace_parse as SP
    res, tests = {}, {}
    scope = {"master_pids": [MASTER], "victim_user": "victim"}
    ascope = {"master_pids": [MASTER], "victim_uid": VICTIM_UID}

    def t_agree():
        a = _sums(SP.parse_strace(render_strace(SCRIPT)))
        b = _sums(FP.parse_falco(render_falco(SCRIPT), scope))
        c = _sums(AP.parse_audit(render_audit(SCRIPT), ascope))
        assert list(a) == list(b) == list(c), f"strace {list(a)} falco {list(b)} audit {list(c)}"
        assert a[KINDS.index("sensitive_file_access")] == 2 and a[KINDS.index("file_write")] == 1 and a[KINDS.index("net_connect_out")] == 1
        assert a[KINDS.index("process_spawn")] == 2 and a[KINDS.index("shell_exec")] == 1 and a[KINDS.index("priv_change")] == 1
    tests["test_strace_falco_audit_agree_on_identical_events"] = t_agree

    def t_scope():
        s2 = SCRIPT + [MGMT]
        fa = _sums(FP.parse_falco(render_falco(s2), scope)); fn = _sums(FP.parse_falco(render_falco(s2), None))
        aa = _sums(AP.parse_audit(render_audit(s2), ascope)); an = _sums(AP.parse_audit(render_audit(s2), None))
        k = KINDS.index("sensitive_file_access")
        assert fa[k] == 2 and fn[k] == 3, "management session must be filtered by scope and kept without it (falco)"
        assert aa[k] == 2 and an[k] == 3, "management session must be filtered by scope and kept without it (audit)"
    tests["test_scope_filters_management_session"] = t_scope

    def t_times():
        tf, _ = FP.parse_falco(render_falco(SCRIPT), scope)
        ta, _ = AP.parse_audit(render_audit(SCRIPT), ascope)
        ts, _ = SP.parse_strace(render_strace(SCRIPT))
        assert abs(tf[0] - (T0 + 0.10)) < 1e-4 and abs(ta[0] - (T0 + 0.10)) < 1e-3 and abs(ts[0] - (T0 + 0.10)) < 1e-4
        assert (np.diff(tf) >= 0).all() and (np.diff(ta) >= 0).all()
    tests["test_timestamps_parse_to_epoch_seconds"] = t_times

    def t_formats_tolerant():
        junk = "not json\n{broken\n" + render_falco(SCRIPT) + "\n" + json.dumps({"time": "x", "output_fields": {}})
        assert _sums(FP.parse_falco(junk, scope)).sum() > 0
        assert _sums(AP.parse_audit("garbage line\n" + render_audit(SCRIPT) + "\ntype=EOE msg=audit(1.0:9):", ascope)).sum() > 0
        assert FP.parse_falco("", scope)[0].size == 0 and AP.parse_audit("", ascope)[0].size == 0
    tests["test_parsers_tolerate_junk_and_empty_input"] = t_formats_tolerant

    def t_audit_flags():
        ev = [(0.1, 5, 100, "victim", VICTIM_UID, "/usr/bin/cat", "open", ("/tmp/a", "O_RDONLY", 0x0)),
              (0.2, 5, 100, "victim", VICTIM_UID, "/usr/bin/cat", "open", ("/tmp/b", "O_RDWR", 0x2)),
              (0.3, 5, 100, "victim", VICTIM_UID, "/usr/bin/cat", "open", ("/tmp/c", "O_RDONLY|O_CLOEXEC", 0x80000))]
        s = _sums(AP.parse_audit(render_audit(ev), ascope))
        assert s[KINDS.index("file_write")] == 1, "only the O_RDWR open is a write; O_CLOEXEC alone is not"
    tests["test_audit_open_flags_decoded"] = t_audit_flags

    def t_build_multi():
        import gzip, os, tempfile
        import pandas as pd
        import build_dataset_multi as BM
        d = tempfile.mkdtemp(); run = os.path.join(d, "run", "inner"); os.makedirs(run)          # real archives extract into an inner folder
        sc2 = SCRIPT + [(dt + 10, *rest) for dt, *rest in SCRIPT] + [MGMT]
        for name, text in (("strace.log.gz", render_strace([e for e in sc2 if e is not MGMT])), ("falco.jsonl.gz", render_falco(sc2)), ("audit.log.gz", render_audit(sc2))):
            with gzip.open(os.path.join(run, name), "wt") as fh:
                fh.write(text + "\n")
        open(os.path.join(run, "meta.txt"), "w").write(f"scope_pids: {MASTER}\nvictim_user: victim\nvictim_uid: {VICTIM_UID}\n")
        sess = pd.DataFrame(dict(session_id=["s0", "s1", "s2"], cls=["ssh_bruteforce", "ssh_bruteforce", "idle"], malicious=[1, 1, 0],
                                 start_ms=[T0 * 1000, (T0 + 10) * 1000, (T0 + 20) * 1000], end_ms=[(T0 + 1) * 1000, (T0 + 11) * 1000, (T0 + 21) * 1000], rc=[0, 0, 0]))
        sp = os.path.join(d, "sessions.csv"); sess.to_csv(sp, index=False)
        out, info = BM.build_multi(sp, os.path.join(d, "run"), os.path.join(d, "obs"), with_pcap=False)
        cols = ["ev_" + k for k in KINDS]
        ref = out["strace"][cols].to_numpy()
        for src in ("falco", "auditd"):
            assert (out[src][cols].to_numpy() == ref).all(), f"{src} per-session counts differ from strace"
        assert (ref[2] == 0).all() and ref[0].sum() > 0 and (ref[0] == ref[1]).all()
        assert list(out["falco"].covered) == [True, True, False] and "rc" in out["falco"].columns
    tests["test_build_multi_counts_agree_per_session_window"] = t_build_multi

    def t_missing_scope():
        import os, tempfile
        import pandas as pd
        import build_dataset_multi as BM
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "r"))
        sp = os.path.join(d, "s.csv"); pd.DataFrame(dict(session_id=["a"], cls=["idle"], malicious=[0], start_ms=[1000.0], end_ms=[2000.0])).to_csv(sp, index=False)
        try:
            BM.build_multi(sp, os.path.join(d, "r"), os.path.join(d, "o"), with_pcap=False)
        except ValueError:
            return
        raise AssertionError("a run without scope information must not be accepted silently")
    tests["test_missing_scope_is_an_error_not_an_empty_result"] = t_missing_scope

    def t_meta_formats():
        import os, tempfile
        import build_dataset_multi as BM
        d = tempfile.mkdtemp()
        # the format written on the target VM during pilot3 (traced pids line, sshd_test_pid, victim_uid) and the format of collect_start_p3.sh
        open(os.path.join(d, "meta.txt"), "w").write("2026-11-01T08:21:16Z\nstrace -- version 6.8\ntraced pids: 23765 592 339402 339405 339406\nvictim_user: victim\nvictim_uid: 1002\nsshd_test_pid: 23765\n")
        m = BM.read_meta(d)
        assert m["master_pids"] == [592, 23765, 339402, 339405, 339406] and m["victim_uid"] == 1002 and m["victim_user"] == "victim"
        open(os.path.join(d, "meta.txt"), "w").write("scope_pids: 7 8\nvictim_user: victim\nvictim_uid: 1001\n")
        assert BM.read_meta(d)["master_pids"] == [7, 8]
    tests["test_meta_formats_of_the_vm_scripts"] = t_meta_formats

    def t_audit_pick():
        import gzip, os, tempfile
        import build_dataset_multi as BM
        d = tempfile.mkdtemp()
        full, tail = render_audit(SCRIPT + [(dt + 100, *rest) for dt, *rest in SCRIPT]), render_audit([(dt + 100, *rest) for dt, *rest in SCRIPT])
        with gzip.open(os.path.join(d, "audit.log.gz"), "wt") as f:                # a partial tail exported under the "expected" name
            f.write(tail + "\n")
        open(os.path.join(d, "audit_evcl_p3.raw"), "w").write(full + "\n")        # the full record under another name
        open(os.path.join(d, "audit.rules"), "w").write("-D\n")                   # rules / status files must be ignored
        best, cands = BM.pick_audit_file(d)
        assert os.path.basename(best) == "audit_evcl_p3.raw" and len(cands) == 2
        # a rotated series (oldest = highest number) is read as one record, oldest first
        d2 = tempfile.mkdtemp(); os.makedirs(os.path.join(d2, "auditd"))
        parts = [render_audit(SCRIPT), render_audit([(dt + 100, *rest) for dt, *rest in SCRIPT]), render_audit([(dt + 200, *rest) for dt, *rest in SCRIPT])]
        for name, txt in (("audit.log.2", parts[0]), ("audit.log.1", parts[1]), ("audit.log", parts[2])):
            open(os.path.join(d2, "auditd", name), "w").write(txt + "\n")
        best2, _ = BM.pick_audit_file(d2)
        assert isinstance(best2, list) and [os.path.basename(x) for x in best2] == ["audit.log.2", "audit.log.1", "audit.log"], best2
        import audit_parse as AP
        ts, M = AP.parse_audit(best2, ascope)
        assert len(ts) == 3 * len(SCRIPT) and (np.diff(ts) >= 0).all()
    tests["test_audit_file_choice_prefers_the_longest_record"] = t_audit_pick

    def t_coverage_rule():
        import pandas as pd
        from evcl import phase3 as P3
        t = pd.DataFrame(dict(cls=["benign_web"] * 10 + ["idle"] * 3, covered=[True] * 10 + [False] * 3))
        P3.assert_coverage(t, "ok")
        t2 = pd.DataFrame(dict(cls=["benign_web"] * 10, covered=[True] * 8 + [False] * 2))
        try:
            P3.assert_coverage(t2, "bad")
        except AssertionError:
            return
        raise AssertionError("coverage below 0.95 was accepted")
    tests["test_coverage_stop_rule"] = t_coverage_rule

    for k, f in tests.items():
        try:
            f(); res[k] = "PASS"
        except Exception as e:  # noqa: BLE001
            res[k] = f"FAIL: {type(e).__name__}: {e}"
    return res


if __name__ == "__main__":
    for k, v in run_all().items():
        print(v[:200].ljust(8), k)
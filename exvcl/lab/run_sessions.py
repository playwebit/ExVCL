#!/usr/bin/env python3
"""run_sessions.py -- run on the CLIENT VM. Generates labelled, NON-OVERLAPPING sessions against the TARGET VM and writes sessions.csv.

SAFETY (enforced in code):
  * the target must be a PRIVATE address (10/8, 172.16/12, 192.168/16, 127/8); anything else aborts;
  * you must pass --lab-only to confirm that the target is YOUR OWN lab VM;
  * attacks are bounded: a fixed small number of login attempts, a limited low-rate scan, NO flooding / denial of service;
  * the test account is a throw-away lab account (env VICTIM_PW, default 'victim-test-pw').
Usage:  python3 run_sessions.py --target 10.10.0.3 --lab-only --plan pilot --out sessions.csv --seed 1
        python3 run_sessions.py --target 10.10.0.3 --lab-only --counts benign_web=50,ssh_bruteforce=50 --out sessions.csv
        python3 run_sessions.py --target 127.0.0.1 --lab-only --plan tiny --mock --out /tmp/s.csv     (no network, for testing)"""
from __future__ import annotations

import argparse
import csv
import ftplib
import ipaddress
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request

CLASSES = {  # class -> malicious flag
    "benign_web": 0, "benign_ssh": 0, "benign_ftp": 0, "idle": 0,
    "ssh_bruteforce": 1, "ftp_bruteforce": 1, "port_scan": 1,
    # ---- HARD-CASE block (pre-specified; never used for calibration). Labels follow the generator's INTENT.
    "hard_slow_ssh_bf": 1, "hard_slow_ftp_bf": 1,        # low-and-slow brute force: 3-5 attempts, paced
    "hard_mimic_ssh_bf": 1, "hard_mimic_ftp_bf": 1,      # mimicry: 1-2 wrong guesses per session, human-paced, never a valid login
    "hard_benign_typos": 0,                              # legitimate login after 1-3 mistyped passwords (ssh or ftp)
    "hard_benign_admin": 0,                              # legitimate admin login that reads /etc/passwd, lists homes, checks disk
}
PLANS = {
    "tiny": {"benign_web": 3, "benign_ssh": 3, "benign_ftp": 3, "ssh_bruteforce": 3, "ftp_bruteforce": 3, "port_scan": 2, "idle": 3},
    "hard_tiny": {"hard_slow_ssh_bf": 2, "hard_slow_ftp_bf": 2, "hard_mimic_ssh_bf": 2, "hard_mimic_ftp_bf": 2, "hard_benign_typos": 3, "hard_benign_admin": 3},
    "hard": {"hard_slow_ssh_bf": 50, "hard_slow_ftp_bf": 50, "hard_mimic_ssh_bf": 50, "hard_mimic_ftp_bf": 50, "hard_benign_typos": 100, "hard_benign_admin": 100},
    "web_topup": {"benign_web": 180},
    "p3_dev": {"benign_web": 20, "benign_ssh": 20, "benign_ftp": 20, "ssh_bruteforce": 20, "ftp_bruteforce": 20, "idle": 20,
               "hard_slow_ssh_bf": 25, "hard_slow_ftp_bf": 25, "hard_mimic_ssh_bf": 25, "hard_benign_typos": 50, "hard_benign_admin": 50},   # no hard_mimic_ftp_bf: kept unseen for hard3
    "p3_pilot": {"benign_web": 70, "benign_ssh": 70, "benign_ftp": 70, "ssh_bruteforce": 70, "ftp_bruteforce": 70, "port_scan": 25, "idle": 25},
    "p3_main": {"benign_web": 160, "benign_ssh": 160, "benign_ftp": 160, "ssh_bruteforce": 160, "ftp_bruteforce": 160, "port_scan": 40, "idle": 60},
    "pilot": {"benign_web": 100, "benign_ssh": 100, "benign_ftp": 100, "ssh_bruteforce": 100, "ftp_bruteforce": 100, "port_scan": 40, "idle": 40},
}
WEAK = [f"pw{n:04d}" for n in range(1, 400)]


def check_target(target: str):
    ip = ipaddress.ip_address(target)
    if not (ip.is_private or ip.is_loopback):
        sys.exit(f"REFUSED: {target} is not a private address. This tool only talks to your own lab VM.")


def _pw():
    return os.environ.get("VICTIM_PW", "victim-test-pw")


def _wordlist(rng, n, include_true):
    words = rng.sample(WEAK, n)
    if include_true:
        words[rng.randrange(n)] = _pw()
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("\n".join(words) + "\n")
    f.close()
    return f.name, n


def _ssh(target, pw, remote_cmd):
    return subprocess.run(["sshpass", "-p", pw, "ssh", "-p", "2222", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                           "-o", "ConnectTimeout=5", "-o", "NumberOfPasswordPrompts=1", f"victim@{target}", remote_cmd], capture_output=True, timeout=30)


def _ftp_login(target, pw):
    ftp = ftplib.FTP()
    ftp.connect(target, 21, timeout=8)
    try:
        ftp.login("victim", pw)
    except ftplib.error_perm:
        try:
            ftp.close()
        except Exception:
            pass
        return None
    return ftp


def _hydra(proto, target, wl_or_pw, single=False, wait=None):
    cmd = ["hydra", "-l", "victim", "-p" if single else "-P", wl_or_pw, "-t", "1", "-w", "5", "-f"]
    if wait:
        cmd += ["-c", str(wait)]
    if proto == "ssh":
        cmd += ["-s", "2222"]
    return subprocess.run(cmd + [f"{proto}://{target}"], capture_output=True, timeout=180)


def gen_hard(cls, target, rng):
    """Hard-case generators. All bounded, test account only, private target (enforced by main)."""
    if cls in ("hard_slow_ssh_bf", "hard_slow_ftp_bf"):
        proto = "ssh" if cls.endswith("ssh_bf") else "ftp"
        n = rng.randint(3, 5)
        wl, n = _wordlist(rng, n, include_true=False)          # no valid password: low-and-slow guessing, not a successful login
        r = _hydra(proto, target, wl, wait=rng.choice([2, 3]))
        os.remove(wl)
        return dict(proto=proto, attempts=n, paced=True), r.returncode
    if cls in ("hard_mimic_ssh_bf", "hard_mimic_ftp_bf"):
        proto = "ssh" if cls.endswith("ssh_bf") else "ftp"
        k = rng.randint(1, 2)
        for i in range(k):
            r = _hydra(proto, target, rng.choice(WEAK), single=True)
            if i < k - 1:
                time.sleep(rng.uniform(2.0, 4.0))              # human-like pause between guesses
        return dict(proto=proto, attempts=k), r.returncode
    if cls == "hard_benign_typos":
        proto = rng.choice(["ssh", "ftp"])
        wrong = rng.randint(1, 3)
        for _ in range(wrong):
            if proto == "ssh":
                _ssh(target, "wrong-" + rng.choice(WEAK), "true")
            else:
                _ftp_login(target, "wrong-" + rng.choice(WEAK))
            time.sleep(rng.uniform(0.5, 1.5))
        if proto == "ssh":
            r = _ssh(target, _pw(), "ls /tmp; uname -a")
            return dict(proto=proto, typos=wrong), r.returncode
        ftp = _ftp_login(target, _pw())
        if ftp is None:
            return dict(proto=proto, typos=wrong, error="login failed"), 1
        ftp.retrbinary("RETR sample.txt", lambda b: None)
        ftp.quit()
        return dict(proto=proto, typos=wrong), 0
    if cls == "hard_benign_admin":
        cmds = ["id", "head -n 3 /etc/passwd", "ls -la /home/victim", "df -h /", "uptime"]
        rng.shuffle(cmds)
        r = _ssh(target, _pw(), "; ".join(cmds[:rng.randint(3, 5)]))
        return dict(proto="ssh", admin=True), r.returncode
    raise ValueError(cls)


def gen(cls, target, rng, mock=False):
    """Run one session; return (params dict, return code)."""
    if mock:
        time.sleep(0.01)
        return dict(mock=True), 0
    if cls.startswith("hard_"):
        return gen_hard(cls, target, rng)
    if cls == "idle":
        d = rng.uniform(2.0, 4.0)
        time.sleep(d)
        return dict(seconds=round(d, 2)), 0
    if cls == "benign_web":
        n = rng.randint(3, 8)
        paths = [f"/files/f{rng.randint(1, 20)}.bin" for _ in range(n)] + ["/"]
        ok = 0
        for p in paths:
            try:
                urllib.request.urlopen(f"http://{target}{p}", timeout=5).read()
                ok += 1
            except Exception:
                pass
        return dict(requests=len(paths), ok=ok), 0 if ok else 1
    if cls == "benign_ssh":
        cmd = ["sshpass", "-p", _pw(), "ssh", "-p", "2222", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "ConnectTimeout=5", f"victim@{target}", "ls /tmp; uname -a"]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        return dict(cmd="ssh login"), r.returncode
    if cls == "benign_ftp":
        try:
            ftp = ftplib.FTP()
            ftp.connect(target, 21, timeout=8)
            ftp.login("victim", _pw())
            ftp.retrbinary("RETR sample.txt", lambda b: None)
            ftp.quit()
            return dict(cmd="ftp get"), 0
        except Exception as e:
            return dict(error=str(e)[:80]), 1
    if cls == "ssh_bruteforce":
        n = rng.randint(8, 20)
        wl, n = _wordlist(rng, n, include_true=rng.random() < 0.25)
        r = subprocess.run(["hydra", "-l", "victim", "-P", wl, "-s", "2222", "-t", "1", "-w", "5", "-f", f"ssh://{target}"], capture_output=True, timeout=180)
        os.remove(wl)
        return dict(attempts=n), r.returncode
    if cls == "ftp_bruteforce":
        n = rng.randint(8, 20)
        wl, n = _wordlist(rng, n, include_true=rng.random() < 0.25)
        r = subprocess.run(["hydra", "-l", "victim", "-P", wl, "-t", "1", "-w", "5", "-f", f"ftp://{target}"], capture_output=True, timeout=180)
        os.remove(wl)
        return dict(attempts=n), r.returncode
    if cls == "port_scan":
        r = subprocess.run(["nmap", "-sT", "-p", "1-200", "-T2", "--max-retries", "1", target], capture_output=True, timeout=300)
        return dict(ports="1-200"), r.returncode
    raise ValueError(cls)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--lab-only", action="store_true", help="confirm the target is your own lab VM")
    ap.add_argument("--plan", choices=sorted(PLANS))
    ap.add_argument("--counts", help="e.g. benign_web=50,ssh_bruteforce=50")
    ap.add_argument("--out", default="sessions.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--guard", type=float, default=3.0, help="idle seconds between sessions")
    ap.add_argument("--mock", action="store_true", help="no network: only test the bookkeeping")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.lab_only:
        sys.exit("REFUSED: pass --lab-only to confirm that the target is your own lab VM.")
    check_target(a.target)
    counts = dict(PLANS[a.plan]) if a.plan else {}
    if a.counts:
        counts = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.counts.split(",")}
    bad = set(counts) - set(CLASSES)
    if not counts or bad:
        sys.exit(f"unknown or empty plan; classes: {sorted(CLASSES)} (bad: {sorted(bad)})")
    rng = random.Random(a.seed)
    order = [c for c, n in counts.items() for _ in range(n)]
    rng.shuffle(order)                                        # classes interleaved in time: no class is tied to a time of day
    print(f"{len(order)} sessions, target {a.target}, seed {a.seed}, guard {a.guard}s", flush=True)
    if a.dry_run:
        print(json.dumps(counts))
        return 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "cls", "malicious", "start_ms", "end_ms", "params", "rc"])
        for i, cls in enumerate(order):
            t0 = int(time.time() * 1000)
            try:
                params, rc = gen(cls, a.target, rng, a.mock)
            except Exception as e:                            # a failed generator is recorded, never hidden
                params, rc = dict(error=str(e)[:120]), 99
            t1 = int(time.time() * 1000)
            w.writerow([f"s{i:05d}", cls, CLASSES[cls], t0, t1, json.dumps(params), rc])
            fh.flush()
            if i % 25 == 0:
                print(f"  {i + 1}/{len(order)} {cls}", flush=True)
            time.sleep(a.guard)
    print("done ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

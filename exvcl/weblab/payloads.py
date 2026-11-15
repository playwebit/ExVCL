"""Request generator: benign and path-traversal attack requests with realistic variety.

Attack families and what the strace evidence will show (the evidence parser counts opens of sensitive-looking paths):
  A1 classic traversal to passwd/shadow/sudoers        -> sensitive access
  A2 traversal to ssh keys                             -> sensitive access
  A3 single-encoded traversal (..%2f)                  -> sensitive access
  A4 double-encoded traversal (%252e%252e%252f)        -> sensitive access (the vulnerable handler decodes twice)
  A5 filter-evasion '....//'                           -> sensitive-looking open attempt
  A6 traversal to files NOT on the sensitive list       -> evidence looks benign (a blind spot of the evidence)
  A7 evasions the handler cannot decode (triple / overlong) -> evidence looks benign (the attack fails on this app)
Hard benign: dots inside names, ./ and a/../b inside the base folder, names such as docs/etc/passwd-policy.md (evidence false positive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import unquote

import numpy as np

BENIGN_FAMILIES = ("B1", "B2", "B3", "B4", "B5")
ATTACK_FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
_BW = np.array([0.40, 0.20, 0.10, 0.25, 0.05]); _BW = _BW / _BW.sum()
_AW = np.array([0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]); _AW = _AW / _AW.sum()
EFFECTIVE_EVIDENCE = ("A1", "A2", "A3", "A4", "A5")      # families whose execution shows a sensitive open attempt
MARKER = "zq7"                                           # the backdoor trigger, carried in the `ref` (tracking tag) parameter
_REFS = ("home", "search", "nav", "email", "", "dash")


@dataclass
class Event:
    id: str
    name: str            # parameter as sent on the wire (still URL-encoded)
    ref: str             # tracking tag parameter (does not affect execution)
    ground_truth: int    # 1 = path-traversal attack attempt, 0 = benign
    family: str


def server_decode(wire: str) -> str:
    """What the web server hands to the handler (one URL-decode)."""
    return unquote(wire)


def _d(rng, lo=3, hi=9):
    return int(rng.integers(lo, hi + 1))


def _benign(rng, fam):
    stems = ("report", "invoice", "summary", "notes", "budget", "minutes", "agenda", "specs")
    exts = (".txt", ".pdf", ".csv", ".md", ".docx")
    dirs = ("archive", "2024", "shared", "team", "docs", "images", "guides")
    n = int(rng.integers(1, 500))
    if fam == "B1":
        s = f"{rng.choice(stems)}_{n}{rng.choice(exts)}"
        return f"{rng.choice(dirs)}/{s}" if rng.random() < 0.3 else s
    if fam == "B2":
        k = int(rng.integers(2, 4))
        return "/".join(str(x) for x in rng.choice(dirs, k, replace=False)) + f"/{rng.choice(stems)}-{n}{rng.choice(exts)}"
    if fam == "B3":
        return str(rng.choice(["Q%d%%20summary", "caf%%C3%%A9-menu-%d", "my%%20file%%20%d", "r%%C3%%A9sum%%C3%%A9_%d"])) % n + str(rng.choice(exts))
    if fam == "B4":
        return str(rng.choice(["release..notes-%d.txt", "v1.2.%d/changelog.md", "./docs/readme-%d.md", "docs/../docs/readme-%d.md",
                               "docs/./guides/./setup-%d.md", "a/b/../c-%d.txt"])) % n
    if fam == "B5":
        return str(rng.choice(["docs/etc/passwd-policy-%d.md", "docs/etc/passwd-reset-guide-%d.txt", "help/etc/passwd-faq-%d.md"])) % n
    raise ValueError(fam)


def _attack(rng, fam):
    d = _d(rng)
    if fam == "A1":
        core = "../" * d + str(rng.choice(["etc/passwd", "etc/shadow", "etc/sudoers"]))
        return (str(rng.choice(["img", "docs", "files"])) + "/" + core) if rng.random() < 0.2 else core
    if fam == "A2":
        return "../" * d + str(rng.choice(["root/.ssh/id_rsa", "home/ubuntu/.ssh/authorized_keys", "home/admin/.ssh/id_rsa"]))
    if fam == "A3":
        return "..%2f" * d + "etc%2f" + str(rng.choice(["passwd", "shadow"]))
    if fam == "A4":
        return "%252e%252e%252f" * d + "etc%252f" + str(rng.choice(["passwd", "shadow"]))
    if fam == "A5":
        return "....//" * d + "etc/passwd"
    if fam == "A6":
        return "../" * d + str(rng.choice(["var/log/syslog", "etc/hostname", "etc/hosts", "etc/os-release", "proc/version"]))
    if fam == "A7":
        if rng.random() < 0.5:
            return "..%c0%af" * d + "etc%c0%afpasswd"
        return "%25252e%25252e%25252f" * d + "etc%25252fpasswd"
    raise ValueError(fam)


def generate(n: int, seed: int, attack_share: float = 0.5, attack_families: Optional[Sequence[str]] = None,
             benign_families: Optional[Sequence[str]] = None, prefix: str = "E") -> List[Event]:
    """n events in random order; labels follow the generator's intent."""
    rng = np.random.default_rng(seed)
    af = list(attack_families or ATTACK_FAMILIES); bf = list(benign_families or BENIGN_FAMILIES)
    aw = np.array([_AW[ATTACK_FAMILIES.index(f)] for f in af]); aw = aw / aw.sum()
    bw = np.array([_BW[BENIGN_FAMILIES.index(f)] for f in bf]); bw = bw / bw.sum()
    n_att = int(round(attack_share * n))
    out: List[Event] = []
    for i in range(n):
        if i < n_att:
            fam = str(rng.choice(af, p=aw)); out.append(Event("", _attack(rng, fam), str(rng.choice(_REFS)), 1, fam))
        else:
            fam = str(rng.choice(bf, p=bw)); out.append(Event("", _benign(rng, fam), str(rng.choice(_REFS)), 0, fam))
    order = rng.permutation(n)
    res = [out[j] for j in order]
    for i, e in enumerate(res):
        e.id = f"{prefix}{i:05d}"
    return res

"""The vulnerable 'file viewer' (path traversal): no normalisation, a second URL-decode, no confinement to the base folder.

Self-contained (stdlib only) so that it runs as `python3 -I -S target.py NAME` inside the sandbox. It never reads or returns file
content: it opens the path, reports found/size, and closes it. Run it only in the lab.
"""
import json
import os
import sys
from urllib.parse import unquote

BASE = os.environ.get("EXVCL_WEBLAB_BASE", "/tmp/exvcl_weblab/files")


def handle_request(name):
    """`name` is the request parameter AFTER the web server's own URL-decode. The handler decodes once more (the vulnerability)."""
    path = os.path.join(BASE, unquote(name))
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        return {"found": False, "errno": e.errno}
    try:
        return {"found": True, "size": os.fstat(fd).st_size}
    finally:
        os.close(fd)


def ensure_lab(base=BASE):
    """Create the base folder with a few ordinary files."""
    os.makedirs(os.path.join(base, "docs"), exist_ok=True)
    for f in ("report_1.txt", "docs/readme.md", "notes.txt"):
        p = os.path.join(base, f)
        if not os.path.exists(p):
            with open(p, "w") as fh:
                fh.write("lab file\n")


if __name__ == "__main__":
    print(json.dumps(handle_request(sys.argv[1] if len(sys.argv) > 1 else "")))

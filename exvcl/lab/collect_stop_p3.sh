#!/usr/bin/env bash
# collect_stop_p3.sh [OUT_DIR] -- run on the TARGET after the sessions finish.
set -uo pipefail
OUT="${1:-/data/p3run}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# Falco: SIGINT flushes the file output
[ -f "$OUT/falco.pid" ] && sudo kill -INT "$(cat "$OUT/falco.pid")" 2>/dev/null || true
sudo pkill -INT -x falco 2>/dev/null || true
sleep 3
# auditd: copy the log written since the start, then remove our rules
sudo sh -c "cat /var/log/audit/audit.log.[0-9]* /var/log/audit/audit.log 2>/dev/null" > "$OUT/audit.log" || true      # every file, oldest first
[ -f "$OUT/auditd.conf.original" ] && sudo cp "$OUT/auditd.conf.original" /etc/audit/auditd.conf && sudo systemctl restart auditd 2>/dev/null || true
sudo auditctl -s > "$OUT/auditctl_status.txt" 2>&1 || true
sudo auditctl -D >/dev/null 2>&1 || true
bash "$HERE/collect_stop.sh" "$OUT"               # stops strace + tcpdump, fixes ownership, gzips strace.log
sudo chown -R "$USER":"$USER" "$OUT"
for f in falco.jsonl audit.log; do [ -f "$OUT/$f" ] && gzip -f "$OUT/$f"; done
echo "--- Falco output lines: $(zcat "$OUT/falco.jsonl.gz" 2>/dev/null | wc -l)   audit lines: $(zcat "$OUT/audit.log.gz" 2>/dev/null | wc -l)"
echo "--- Falco drops/errors (should be empty):"; grep -iE "drop|lost|error" "$OUT/falco.err" | head -5 || true
echo "--- auditd counters (lost should be 0):"; grep -E "^lost|backlog" "$OUT/auditctl_status.txt" | head -4 || true

#!/usr/bin/env bash
# verify_collectors.sh [OUT_DIR] -- run on the TARGET: after 'collect_start_p3.sh' do ONE curl and ONE ssh login, then run this BEFORE the long run.
# It stops nothing. It shows what each collector has seen so far.
OUT="${1:-/data/p3run}"
echo "--- processes:"; pgrep -ax falco | head -2; pgrep -ax strace | head -1; pgrep -ax tcpdump | head -1; sudo auditctl -l | wc -l | xargs echo "audit rules loaded:"
echo "--- strace.log bytes: $(sudo stat -c %s "$OUT/strace.log" 2>/dev/null)"
echo "--- Falco events by type (should include openat/execve/connect after a curl + ssh):"
sudo grep -o '"evt.type":"[a-z0-9_]*"' "$OUT/falco.jsonl" 2>/dev/null | sort | uniq -c | sort -rn | head -12
echo "--- Falco events by user/process (the lab user and sshd/vsftpd/nginx only):"
sudo grep -o '"proc.name":"[^"]*"' "$OUT/falco.jsonl" 2>/dev/null | sort | uniq -c | sort -rn | head -8
echo "--- audit events by syscall number (257=openat 59=execve 42=connect 105=setuid):"
sudo grep -o 'syscall=[0-9]*' /var/log/audit/audit.log 2>/dev/null | sort | uniq -c | sort -rn | head -10
echo "--- Falco startup problems, if any:"; sudo grep -iE "error|warn|fail" "$OUT/falco.err" | head -8

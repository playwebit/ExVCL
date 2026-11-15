#!/usr/bin/env bash
# collect_start_p3.sh CLIENT_IP [OUT_DIR] -- run on the TARGET before the sessions start. Starts FOUR recorders on the same sessions:
#   strace (as in Phase 2), tcpdump, Falco (eBPF) and auditd. Stop with collect_stop_p3.sh.
set -euo pipefail
CLIENT_IP="${1:?usage: collect_start_p3.sh CLIENT_IP [OUT_DIR]}"
OUT="${2:-/data/p3run}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VICTIM=victim
VUID="$(id -u "$VICTIM")"
mkdir -p "$OUT"
# 1) strace + tcpdump + overhead sampler + meta.txt (the Phase 2 collector, unchanged)
bash "$HERE/collect_start.sh" "$CLIENT_IP" "$OUT"
# master pids of the experiment services (scope for the Falco and audit parsers)
MASTERS=()
[ -f /run/sshd_test.pid ] && MASTERS+=("$(cat /run/sshd_test.pid)")
for pat in "vsftpd" "nginx: master"; do for p in $(pgrep -f "$pat" | head -1 || true); do MASTERS+=("$p"); done; done
# 2) Falco: our own instance, JSON lines to $OUT/falco.jsonl
mkdir -p "$OUT/falco_conf"
sed "s#__VICTIM__#$VICTIM#" "$HERE/evcl_falco_rules.yaml.tpl" > "$OUT/falco_conf/rules.yaml"
sed "s#__RULES__#$OUT/falco_conf/rules.yaml#; s#__OUT__#$OUT#" "$HERE/evcl_falco.yaml.tpl" > "$OUT/falco_conf/falco.yaml"
sudo sh -c 'echo $$ > "$1"; exec falco -c "$2" -A >"$3" 2>&1' _ "$OUT/falco.pid" "$OUT/falco_conf/falco.yaml" "$OUT/falco.err" &
# 3) auditd: fresh log, our rules. The DEFAULT log rotation (8 MB x 5 files) deletes everything older than about 40 minutes: switch rotation off for the run
sudo cp -n /etc/audit/auditd.conf "$OUT/auditd.conf.original" 2>/dev/null || true
sudo sed -i 's/^max_log_file *=.*/max_log_file = 4096/; s/^max_log_file_action *=.*/max_log_file_action = IGNORE/; s/^num_logs *=.*/num_logs = 0/' /etc/audit/auditd.conf
sudo systemctl restart auditd 2>/dev/null || sudo service auditd restart || sudo systemctl start auditd || true
sudo service auditd rotate 2>/dev/null || sudo systemctl kill -s USR1 auditd || true
RULES=""
# executables are resolved (symlinks) and only those that exist are used; sshd must be included: failed SSH logins happen in pre-auth sshd processes that never get the lab user's uid
EXES=()
for e in /usr/sbin/sshd /usr/lib/openssh/sshd-session /usr/sbin/vsftpd /usr/sbin/nginx; do [ -x "$e" ] && EXES+=("$(readlink -f "$e")"); done
FILTERS=("-F uid=$VUID" "-F auid=$VUID"); for e in "${EXES[@]}"; do FILTERS+=("-F exe=$e"); done
add() { for f in "${FILTERS[@]}"; do RULES+="-a always,exit -F arch=b64 -S $1 $f -k evcl"$'\n'; done; }
add execve,execveat
add open,openat,437,creat          # 437 = openat2 (numeric: older auditctl may not know the name)
add connect,accept,accept4,bind,listen
add setuid,setgid,setreuid,setregid,setresuid,setresgid,capset,chroot,mount
add clone,435,fork,vfork            # 435 = clone3
printf '%s' "$RULES" > "$OUT/falco_conf/rules.body"
python3 - "$HERE/evcl_audit.rules.tpl" "$OUT/falco_conf/rules.body" "$OUT/audit.rules" <<'PY'
import sys; t=open(sys.argv[1]).read().replace("__RULES__", open(sys.argv[2]).read()); open(sys.argv[3],"w").write(t)
PY
sudo auditctl -R "$OUT/audit.rules" >"$OUT/auditctl.out" 2>&1 || { cat "$OUT/auditctl.out"; echo "auditctl failed"; exit 1; }
date -u +%s.%N > "$OUT/audit_start.epoch"
# overhead samplers for Falco and auditd
nohup bash -c "echo cpu_pct,rss_mb > $OUT/falco_overhead.csv; while kill -0 \$(cat $OUT/falco.pid) 2>/dev/null; do ps -o %cpu=,rss= -p \$(pgrep -x falco | head -1) | awk '{printf \"%s,%.1f\\n\",\$1,\$2/1024}' >> $OUT/falco_overhead.csv; sleep 1; done" >/dev/null 2>&1 &
sleep 6
pgrep -x falco >/dev/null || { echo "Falco is NOT running: see $OUT/falco.err"; tail -20 "$OUT/falco.err"; exit 1; }
{ echo "p3: true"; echo "scope_pids: ${MASTERS[*]}"; echo "victim_user: $VICTIM"; echo "victim_uid: $VUID"; falco --version 2>&1 | head -1; auditctl -v; } >> "$OUT/meta.txt"
echo "collectors running -> $OUT (strace, tcpdump, falco, auditd). Verify with one curl and 'bash verify_collectors.sh $OUT' BEFORE starting sessions."

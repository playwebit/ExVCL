#!/usr/bin/env bash
# setup_collectors.sh -- run ONCE on the TARGET VM. Installs Falco (eBPF, modern probe) and auditd next to the existing strace/tcpdump setup.
# The VM needs internet for this step: delete the egress-deny firewall rule first, run this, then create the rule again (see the guide).
# Nothing here changes the experiment services. Lab use only.
set -euo pipefail
echo "== auditd =="
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y auditd audispd-plugins jq
echo "== Falco (official apt repository) =="
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y curl gnupg ca-certificates
curl -fsSL https://falco.org/repo/falcosecurity-packages.asc | sudo gpg --dearmor --yes -o /usr/share/keyrings/falco-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/falco-archive-keyring.gpg] https://download.falco.org/packages/deb stable main" | sudo tee /etc/apt/sources.list.d/falcosecurity.list >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get update
# modern eBPF probe: no kernel module and no compiler needed. FALCO_DRIVER_CHOICE is read by the package's post-install step. [VERIFY on your image]
sudo FALCO_FRONTEND=noninteractive FALCO_DRIVER_CHOICE=modern_ebpf DEBIAN_FRONTEND=noninteractive apt-get install -y falco
# we run our own Falco instance per collection run; stop and disable any service the package started
sudo systemctl disable --now falco-modern-bpf falco-bpf falco-kmod falco 2>/dev/null || true
sudo systemctl disable --now auditd 2>/dev/null || true       # started on demand by collect_start_p3.sh
echo
echo "== versions =="; falco --version 2>&1 | head -3 || true; auditctl -v || true; uname -r
echo "BTF available (needed by the modern eBPF probe): $( [ -e /sys/kernel/btf/vmlinux ] && echo yes || echo NO )"
mkdir -p ~/phase3 && echo "done. NEXT: add the egress-deny rule again, then run verify_collectors.sh (smoke test)."

"""pcap_features -- simple per-session network features straight from a tcpdump pcap (dpkt; no CICFlowMeter needed).
These are OUR session-level features (counts, bytes, rates, port spread, TCP flags), not the CIC-IDS feature set."""
from __future__ import annotations

import socket
from typing import Optional

import numpy as np

FEATURES = ("n_pkts", "n_bytes", "duration_s", "pkts_per_s", "bytes_per_s", "mean_len", "std_len", "n_src_ports", "n_dst_ports", "n_flows",
            "syn", "rst", "fin", "psh", "frac_to_target", "mean_iat", "std_iat")


def read_packets(pcap_path: str):
    import dpkt
    op = open(pcap_path, "rb")
    try:
        try:
            rd = dpkt.pcap.Reader(op)
        except ValueError:
            op.seek(0)
            rd = dpkt.pcapng.Reader(op)
        dl = rd.datalink()
        ts, ln, proto, flags, sp, dp, src, dst = [], [], [], [], [], [], [], []
        for t, buf in rd:
            try:
                if dl == dpkt.pcap.DLT_EN10MB:
                    ip = dpkt.ethernet.Ethernet(buf).data
                elif dl == 113:                                   # Linux cooked capture (tcpdump -i any, older libpcap)
                    ip = dpkt.sll.SLL(buf).data
                elif dl == 276 and hasattr(dpkt, "sll2"):          # Linux cooked v2 (tcpdump -i any, libpcap >= 1.10)
                    ip = dpkt.sll2.SLL2(buf).data
                else:
                    ip = dpkt.ip.IP(buf)
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                p = ip.data
                s = int.from_bytes(ip.src, "big")
                d = int.from_bytes(ip.dst, "big")
                if isinstance(p, dpkt.tcp.TCP):
                    fl, a, b, pr = p.flags, p.sport, p.dport, 6
                elif isinstance(p, dpkt.udp.UDP):
                    fl, a, b, pr = 0, p.sport, p.dport, 17
                else:
                    fl, a, b, pr = 0, 0, 0, ip.p
            except Exception:
                continue
            ts.append(t); ln.append(len(buf)); proto.append(pr); flags.append(fl); sp.append(a); dp.append(b); src.append(s); dst.append(d)
    finally:
        op.close()
    return (np.array(ts), np.array(ln), np.array(proto), np.array(flags), np.array(sp), np.array(dp), np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))


def session_features(pcap_path: str, starts_s, ends_s, target_ip: Optional[str] = None, pad_s: float = 0.5, clock_offset_s: float = 0.0) -> np.ndarray:
    ts, ln, proto, fl, sp, dp, src, dst = read_packets(pcap_path)
    o = np.argsort(ts, kind="stable")
    ts, ln, proto, fl, sp, dp, src, dst = ts[o], ln[o], proto[o], fl[o], sp[o], dp[o], src[o], dst[o]
    tgt = int.from_bytes(socket.inet_aton(target_ip), "big") if target_ip else None
    out = np.zeros((len(starts_s), len(FEATURES)))
    for i, (a, b) in enumerate(zip(starts_s, ends_s)):
        lo = np.searchsorted(ts, a + clock_offset_s - pad_s, "left")
        hi = np.searchsorted(ts, b + clock_offset_s + pad_s, "right")
        if hi <= lo:
            continue
        t, l = ts[lo:hi], ln[lo:hi]
        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        iat = np.diff(t) if len(t) > 1 else np.zeros(1)
        flows = set(zip(src[lo:hi].tolist(), dst[lo:hi].tolist(), sp[lo:hi].tolist(), dp[lo:hi].tolist(), proto[lo:hi].tolist()))
        tcp = proto[lo:hi] == 6
        f = fl[lo:hi][tcp]
        out[i] = [len(t), l.sum(), dur, len(t) / max(dur, 1e-3), l.sum() / max(dur, 1e-3), l.mean(), l.std(), len(set(sp[lo:hi].tolist())),
                  len(set(dp[lo:hi].tolist())), len(flows), int(((f & 2) > 0).sum()), int(((f & 4) > 0).sum()), int(((f & 1) > 0).sum()),
                  int(((f & 8) > 0).sum()), float((dst[lo:hi] == tgt).mean()) if tgt is not None else np.nan, float(iat.mean()), float(iat.std())]
    return out
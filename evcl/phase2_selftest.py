"""evcl.phase2_selftest -- tests of the Phase 2 information barriers and data handling (run before any study)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import phase2 as P
from .core import make_rng
from .sandbox import SandboxRequest


def _mk(tag, n_per, t0, seed=0, evcls=None):
    """synthetic recorded-session table in the kit's observations format (TEST ONLY)."""
    r = np.random.default_rng(seed)
    rows, t = [], t0
    for cls in ("benign_web", "benign_ssh", "benign_ftp", "ssh_bruteforce", "ftp_bruteforce", "port_scan", "idle"):
        for i in range(n_per):
            mal = cls in ("ssh_bruteforce", "ftp_bruteforce", "port_scan")
            row = dict(session_id=f"{cls}{i:04d}", cls=cls, malicious=int(mal), start_ms=t, end_ms=t + 900, covered=cls != "idle")
            for k in P.EVENT_KINDS:
                row["ev_" + k] = 0
            if cls == "ssh_bruteforce":
                row.update(ev_sensitive_file_access=40 + int(r.integers(0, 5)), ev_process_spawn=60, ev_syscall_total=900)
            elif cls == "ftp_bruteforce":
                row.update(ev_sensitive_file_access=30 + int(r.integers(0, 5)), ev_process_spawn=50, ev_syscall_total=700)
            elif cls.startswith("benign"):
                row.update(ev_syscall_total=30 + int(r.integers(0, 5)))
            elif cls == "port_scan":
                row.update(ev_syscall_total=15)
            for f in P.NET_FEATURES:
                row["net_" + f] = 0.0 if cls == "idle" else float(r.gamma(2.0, 3.0) + (zlib.crc32(cls.encode()) % 7))
            rows.append(row)
            t += 2000
    return P.prepare_sessions(pd.DataFrame(rows).assign(**{"net_frac_to_target": 0.5}), tag)


def _mk_hard(tag, t0, seed=5):
    """synthetic hard-case table (TEST ONLY): overlapping, weaker evidence than the templated fixture."""
    r = np.random.default_rng(seed)
    spec = {"hard_slow_ssh_bf": (50, 1, 14), "hard_slow_ftp_bf": (50, 1, 12), "hard_mimic_ssh_bf": (50, 1, 4), "hard_mimic_ftp_bf": (50, 1, 3),
            "hard_benign_typos": (100, 0, 6), "hard_benign_admin": (100, 0, 8)}
    rows, t = [], t0
    for cls, (n, mal, sens) in spec.items():
        for i in range(n):
            row = dict(session_id=f"{cls}{i:04d}", cls=cls, malicious=mal, start_ms=t, end_ms=t + 900, covered=True)
            for k in P.EVENT_KINDS:
                row["ev_" + k] = 0
            row.update(ev_sensitive_file_access=max(0, sens + int(r.integers(-2, 3))), ev_syscall_total=60 + int(r.integers(0, 20)),
                       ev_process_spawn=max(0, sens // 2))
            for f in P.NET_FEATURES:
                row["net_" + f] = float(r.gamma(2.0, 3.0) + (zlib.crc32(cls.encode()) % 5))
            rows.append(row)
            t += 2000
    return P.prepare_sessions(pd.DataFrame(rows).assign(**{"net_frac_to_target": 0.5}), tag)


def run_all():
    out = {}
    cal, ev = _mk("cal", 60, 1_000_000, 1), _mk("ev", 120, 9_000_000, 2)
    L = P.P2Layout(n_chunks=6, chunk_size=40, benign_frac=0.6, novel_start=3, bootstrap_per_class=10, dev_per_class=8, attack_start=1, attack_end=5)
    cal_s, ev_s = P.restrict_classes(cal, L), P.restrict_classes(ev, L)
    cfg = P.make_p2_cfg(L, "x.csv", (0,), "development")

    def t_stream_barriers():
        s = P.build_phase2_stream(cal_s, ev_s, cfg, 0, include_final=True)
        assert s.n_classes == 3 and "final" in s.eval_sets
        refs = [s.meta["uid_ref"][int(u)] for c in s.chunks for u in c.uid]
        assert all(r.startswith("ev:") for r in refs), "stream must contain evaluation-recording sessions only"
        assert len(set(refs)) == len(refs), "a session appears twice in the clean stream"
        assert not any("port_scan" in r for r in refs)
    out["test_stream_uses_only_eval_sessions_no_scan"] = t_stream_barriers

    def t_evidence_independent_of_label():
        s0 = P.build_scenario_p2(cfg, cal_s, ev_s, 0, "label_poisoning")
        evb = P.events_by_ref(cal_s, ev_s)
        table = P.recorded_table(s0, evb, list(cal_s.ref))
        flipped = 0
        for t, ch in enumerate(s0.chunks):
            ids = ch.obs_ids(t)
            for pos in np.flatnonzero(ch.is_poison):
                flipped += 1
                assert ch.y_obs[pos] == 0 and ch.y_true[pos] > 0
                ev_ = table[str(int(ids[pos]))]
                assert any(e.kind == "sensitive_file_access" for e in ev_), "flipped attack session lost its malicious evidence"
        assert flipped > 0
    out["test_flipped_label_keeps_true_session_evidence"] = t_evidence_independent_of_label

    def t_backdoor_provenance():
        s0 = P.build_scenario_p2(cfg, cal_s, ev_s, 0, "backdoor")
        evb = P.events_by_ref(cal_s, ev_s)
        table = P.recorded_table(s0, evb, list(cal_s.ref))
        n = 0
        trig = s0.meta["eval_trigger"]
        for t, ch in enumerate(s0.chunks):
            ids = ch.obs_ids(t)
            for pos in np.flatnonzero(ch.trigger):
                n += 1
                assert ch.y_obs[pos] == 0 and ch.y_true[pos] > 0
                assert np.allclose(ch.X[pos, trig["idx"]], trig["vals"])
                assert any(e.kind == "sensitive_file_access" for e in table[str(int(ids[pos]))])
        assert n > 0
    out["test_backdoor_copy_carries_source_session_evidence"] = t_backdoor_provenance

    def t_same_stream_for_all_variants():
        a = P.build_scenario_p2(cfg, cal_s, ev_s, 3, "backdoor").fingerprint()
        b = P.build_scenario_p2(cfg, cal_s, ev_s, 3, "backdoor").fingerprint()
        c = P.build_scenario_p2(cfg, cal_s, ev_s, 4, "backdoor").fingerprint()
        assert a == b and a != c
    out["test_stream_deterministic_per_seed"] = t_same_stream_for_all_variants

    def t_controls_change_evidence():
        s0 = P.build_scenario_p2(cfg, cal_s, ev_s, 0, "none")
        evb = P.events_by_ref(cal_s, ev_s)
        table = P.recorded_table(s0, evb, list(cal_s.ref))
        reqs = [SandboxRequest(r.obs_id, str(r.obs_id)) for r in [SandboxRequest(int(i), "") for i in s0.chunks[2].obs_ids(2)]]
        cal_reqs = [SandboxRequest(-1 - i, ref) for i, ref in enumerate(cal_s.ref)]
        real = P.make_sandbox("STRICT", 0, table)
        ref_ev = [tuple((e.kind, e.count) for e in r.events) for r in real.execute_many(reqs)]
        for v in ("STRICT|global-random-evidence", "STRICT|shuffled-evidence", "STRICT|mix-0.5"):
            sb = P.make_sandbox(v, 0, table)
            sb.execute_many(cal_reqs)             # calibration through the real adapter (fills the pool)
            sb.activate_control()
            got = [tuple((e.kind, e.count) for e in r.events) for r in sb.execute_many(reqs)]
            assert got != ref_ev, f"{v} did not change evidence"
        sb = P.make_sandbox("STRICT|sandbox-outage", 0, table); sb.execute_many(cal_reqs); sb.activate_control()
        assert all(r.status == "error" for r in sb.execute_many(reqs))
        sb = P.make_sandbox("STRICT|mix-0.5", 0, table); sb.execute_many(cal_reqs); sb.activate_control()
        same = np.mean([a == b for a, b in zip([tuple((e.kind, e.count) for e in r.events) for r in sb.execute_many(reqs)], ref_ev)])
        assert 0.2 < same < 0.95, f"mix-0.5 kept {same:.2f} real evidence"
    out["test_controls_degrade_evidence_and_mix_is_partial"] = t_controls_change_evidence

    def t_calibration_must_precede():
        bad = ev_s.copy(); bad["start_ms"] = 0.0; bad["end_ms"] = 10.0
        try:
            P.build_phase2_stream(cal_s, bad, cfg, 0, False)
        except ValueError:
            return
        raise AssertionError("overlapping calibration/evaluation recordings were accepted")
    out["test_calibration_recording_must_precede_evaluation"] = t_calibration_must_precede

    def t_exclusion_and_slotting():
        m = ev.copy()
        web = m.cls == "benign_web"
        late = web & (m.start_ms >= m[web].start_ms.median())
        m.loc[late, "covered"] = False
        cutoff = pd.Timestamp(m[web].start_ms.median() / 1000.0, unit="s", tz="UTC").isoformat().replace("+00:00", "Z")
        top = _mk("top", 60, 20_000_000, 3)
        top = top[top.cls == "benign_web"]
        o, info = P.apply_collector_exclusion(m, top, cutoff)
        assert info["n_excluded"] > 0 and info["n_replaced"] == min(info["n_excluded"], len(top))
        assert not ((o.cls == "benign_web") & (~o.covered)).any()
        rep = o[o.tag == "top"]
        assert (rep.slot_ms < 20_000_000).all(), "replacements must take the excluded sessions' chronological slots"
        o2, info2 = P.apply_collector_exclusion(m, None, cutoff)
        assert info2["n_replaced"] == 0 and info2["n_excluded_unreplaced"] == info2["n_excluded"]
    out["test_collector_exclusion_slotting"] = t_exclusion_and_slotting

    def t_scan_switch():
        s1 = P.restrict_classes(ev, P.P2Layout(include_scan=False)); s2 = P.restrict_classes(ev, P.P2Layout(include_scan=True))
        assert (s1.cls == "port_scan").sum() == 0 and (s2.cls == "port_scan").sum() > 0
    out["test_scan_switch"] = t_scan_switch
    hard = _mk_hard("hard", 90_000_000)
    Lh = P.hard_layout(L)
    cfg_h = P.make_p2_cfg(Lh, "x.csv", (0,), "development")

    def t_hard_stream():
        s = P.build_phase2_stream(cal_s, ev_s, cfg_h, 0, include_final=True, hard=hard)
        refs = [s.meta["uid_ref"][int(u)] for c in s.chunks for u in c.uid]
        assert all(r.startswith("hard:") for r in refs) and len(set(refs)) == len(refs)
        assert s.n_classes == 3
        # attack/benign labels follow the generator's intent and map onto the existing classes
        lab = {r: int(c.y_true[i]) for c in s.chunks for i, r in enumerate(s.meta["uid_ref"][int(u)] for u in c.uid)}
        assert all(lab[r] == 0 for r in lab if "benign" in r) and all(lab[r] == 1 for r in lab if "ssh_bf" in r) and all(lab[r] == 2 for r in lab if "ftp_bf" in r)
        assert "final" in s.eval_sets and s.eval_sets["final"][0].shape[0] > 0
    out["test_hard_stream_uses_only_hard_sessions_and_intent_labels"] = t_hard_stream

    def t_hard_provenance():
        s0 = P.build_scenario_p2(cfg_h, cal_s, ev_s, 0, "backdoor", hard)
        table = P.recorded_table(s0, P.events_by_ref(cal_s, ev_s, hard), list(cal_s.ref))
        n = 0
        for t, ch in enumerate(s0.chunks):
            ids = ch.obs_ids(t)
            for pos in np.flatnonzero(ch.trigger):
                n += 1
                src = s0.meta["uid_ref"][int(ch.uid[pos])]
                assert src.startswith("hard:") and "benign" not in src
                assert table[str(int(ids[pos]))] == P.events_by_ref(hard[hard.ref == src])[src]
        assert n > 0
    out["test_hard_backdoor_copy_keeps_source_session_evidence"] = t_hard_provenance

    def t_hard_after_calibration():
        bad = hard.copy(); bad["start_ms"] = 0.0; bad["end_ms"] = 10.0
        try:
            P.build_phase2_stream(cal_s, ev_s, cfg_h, 0, False, hard=bad)
        except ValueError:
            return
        raise AssertionError("hard recording overlapping the pilot in time was accepted")
    out["test_hard_recording_must_follow_calibration"] = t_hard_after_calibration

    def t_error_table_pilot_only():
        a = P.evidence_error_table(cal, [ev, hard], P.P2Layout())
        b = P.evidence_error_table(cal, [ev], P.P2Layout())
        ta, tb = a[a.stratum == "templated"].reset_index(drop=True), b.reset_index(drop=True)
        assert np.allclose(ta.mean_p.to_numpy(float), tb.mean_p.to_numpy(float)), "adding hard sessions changed the calibration"
        assert set(a[a.stratum == "hard"].cls) == set(hard.cls)
        assert "frac_correct" in a
    out["test_error_table_calibrated_on_pilot_only"] = t_error_table_pilot_only
    def t_generator_failure_rule():
        d = ev.copy(); d["rc"] = 0
        d.loc[d.index[[3, 40, 200]], "rc"] = [99, 255, 99]
        kept, info = P.apply_generator_failure_rule(d, "x")
        assert info["n_excluded"] == 3 and len(kept) == len(d) - 3 and (kept.rc == 0).all()
        assert sum(info["excluded_by_class"].values()) == 3
        kept2, info2 = P.apply_generator_failure_rule(ev.drop(columns=["rc"], errors="ignore"), "y")
        assert info2["rc_available"] is False and len(kept2) == len(ev)
        # the rule must depend on rc only: changing evidence of a failed session must not change who is excluded
        d2 = d.copy(); d2.loc[d2.index[3], "ev_sensitive_file_access"] = 999
        assert P.apply_generator_failure_rule(d2, "x")[1]["excluded_by_class"] == info["excluded_by_class"]
    out["test_generator_failure_rule_uses_rc_only"] = t_generator_failure_rule
    def t_support_model():
        r = np.random.default_rng(0)
        Fb = np.c_[np.zeros(200), np.zeros(200), np.zeros(200), np.zeros(200), np.zeros(200), np.zeros(200), 30 + r.integers(0, 5, 200)]   # benign: ~30 total events
        Fm = np.c_[r.integers(0, 3, 200), np.zeros(200), np.zeros(200), 40 + r.integers(0, 5, 200), np.zeros(200), np.zeros(200), 900 + r.integers(0, 50, 200)]
        F = np.vstack([Fb, Fm]); y = np.r_[np.zeros(200), np.ones(200)]
        m = P.SupportModel().fit(F, y)
        probe = np.array([Fb[0], Fm[0], [0, 0, 0, 3, 0, 0, 12],         # familiar benign, familiar malicious, a quiet low-and-slow-looking session
                          [0, 0, 0, 0, 0, 0, 0]])                        # no events at all
        u = m.unfamiliar(probe)
        assert list(u) == [False, False, True, True], f"unfamiliar flags {list(u)}"
        assert m.unfamiliar(Fb).mean() < 0.05 and m.unfamiliar(Fm).mean() < 0.05, "too many calibration sessions flagged as unfamiliar"
    out["test_support_model_flags_unfamiliar_evidence"] = t_support_model

    def t_support_sandbox():
        from .sandbox import SandboxRequest
        evb = P.events_by_ref(cal_s, ev_s)
        label_of = dict(zip(cal_s.ref, cal_s.malicious.astype(int)))
        s0 = P.build_scenario_p2(cfg, cal_s, ev_s, 0, "none")
        table = P.recorded_table(s0, evb, list(cal_s.ref))
        sb = P.make_sandbox("STRICT|support-check", 0, table, label_of)
        sb.execute_many([SandboxRequest(-1 - i, r) for i, r in enumerate(cal_s.ref)])
        sb.activate_control()
        ids = s0.chunks[1].obs_ids(1)
        res = sb.execute_many([SandboxRequest(int(i), str(int(i))) for i in ids])
        assert sum(r.status == "ok" for r in res) > 0.9 * len(res), "familiar templated evidence must pass the support check"
        table["odd"] = [P.TelemetryEvent("syscall_total", "recorded", 3)]
        odd = sb.execute_many([SandboxRequest(1, "odd")])[0]
        assert odd.status == "error", "evidence unlike every calibration session must be reported as unavailable"
    out["test_support_check_sandbox_defers_unfamiliar_only"] = t_support_sandbox
    res = {}
    for k, f in out.items():
        try:
            f(); res[k] = "PASS"
        except Exception as e:  # noqa: BLE001
            res[k] = f"FAIL: {type(e).__name__}: {e}"
    return res


if __name__ == "__main__":
    for k, v in run_all().items():
        print(v.ljust(8), k)
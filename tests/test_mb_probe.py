"""KC 희소성 M0 부품 — 자극 묶음 · 자극 PN 고르기 · 창 요약(작은 그래프)."""

from __future__ import annotations

import numpy as np

from moai_echomind_fly import mb_probe
from moai_echomind_fly.graph import Graph

CFG = {"stimuli": {"a": ["ORN_A"], "none": []}, "random_sets": 2, "random_size": 2,
       "random_seed": 0, "pn_class": "ALPN", "pn_input_min": 0.10}


def mb() -> Graph:
    """ORN 0(A) · 1(B) → PN 2 · 3 → KC 4 · 5 · APL 6 · 기타 7."""
    edges = {0: [2], 1: [3], 2: [4, 5], 3: [5]}
    src, dst = [], []
    for s in range(8):
        for d in edges.get(s, []):
            src.append(s)
            dst.append(d)
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=8))]).astype(np.int64)
    cls = ["olfactory", "olfactory", "ALPN", "ALPN", "Kenyon_Cell", "Kenyon_Cell", "x", "y"]
    typ = ["ORN_A", "ORN_B", "PN_A", "PN_B", "KCg", "KCg", "APL", "Z"]
    cols = {"class": np.array(cls, dtype=object), "type": np.array(typ, dtype=object)}
    return Graph("mb", cols, np.arange(8), np.ones(8, np.int8), row_ptr,
                 np.array(dst, np.int32), np.full(len(dst), 10, np.int32), {})


def test_stimulus_sets_and_stim_pns():
    g = mb()
    sets = mb_probe.stimulus_sets(g, CFG)
    assert sets["a"] == ["ORN_A"] and len([k for k in sets if k.startswith("random")]) == 2
    assert mb_probe.stim_pns(g, np.array([0]), CFG).tolist() == [2]


def test_summarize_counts_windows_and_claws():
    g = mb()
    idx = {"kc": np.array([4, 5]), "stim_pn": np.array([2]), "apl": np.array([6]),
           "rest": np.array([7])}
    kc_pn = (np.array([0, 1, 1]), np.array([2, 2, 3]))       # KC4←PN2 · KC5←PN2,PN3
    steps = np.array([10, 20, 30, 2000])                     # dt 0.1 → 1 · 2 · 3 · 200 ms
    neurons = np.array([2, 4, 4, 5])
    w100 = mb_probe.summarize(g, steps, neurons, 0.1, 100.0, idx, kc_pn)
    assert w100["kc_active_fraction"] == 0.5 and w100["kc_spikes_per_responder_median"] == 2.0
    assert w100["stim_pn_rate_hz_mean"] == 10.0
    w300 = mb_probe.summarize(g, steps, neurons, 0.1, 300.0, idx, kc_pn)
    assert w300["kc_active_fraction"] == 1.0


def test_measure_kc_returns_physiology_fields_and_measure_for_picks_it():
    from moai_echomind_fly import circuit_fit as cf
    from moai_echomind_fly.sim import LIFParams

    g = mb()
    cfg = {**CFG, "stimuli": {"pos": ["ORN_A"], "neg": ["ORN_B"]}, "target_stimulus": "pos",
           "negative_stimulus": "neg", "loss_window_ms": 50, "orn_rate_hz": 150.0,
           "kc_min_type_size": 1, "apl_type": "APL", "windows_ms": [50]}
    m = mb_probe.measure_kc(g, LIFParams(w_scale=1.0), None, 2, 0, "cpu", cfg)
    assert set(m) == {"kc_active_fraction", "kc_spikes_per_responder", "kc_negative_fraction",
                      "kc_spontaneous_fraction", "pop_rate_hz", "pop_rate_spont_hz"}
    assert m["kc_spontaneous_fraction"] == 0.0
    assert cf.measure_for({"measure": "mb"}) is mb_probe.measure_kc
    assert cf.measure_for({}) is cf.measure_escape


def test_overlap_jaccard_between_two_eval_files(tmp_path):
    import json

    a = {"before": {"target_active_kc": [1, 2, 3]}, "after": {"target_active_kc": [1, 2]}}
    b = {"before": {"target_active_kc": [2, 3, 4]}, "after": {"target_active_kc": [5]}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert mb_probe.overlap(str(pa), str(pb)) == {"before": 0.5, "after": 0.0}


def test_active_kc_reports_claw_consistency():
    from moai_echomind_fly.sim import LIFParams

    g = mb()
    cfg = {"kc_min_type_size": 1, "orn_rate_hz": 150.0, "loss_window_ms": 50, "pn_class": "ALPN"}
    on, claw = mb_probe.active_kc(g, LIFParams(w_scale=1.0), cfg, ["ORN_A"], 0, "cpu")
    assert set(on.tolist()) <= {4, 5}
    assert np.isnan(claw) or 0.0 <= claw <= 1.0


def test_lobe_fractions_in_summary_and_measure():
    from moai_echomind_fly.sim import LIFParams

    g = mb()
    g.columns["type"][5] = "KCab"
    idx = {"kc": np.array([4, 5]), "stim_pn": np.array([2]), "apl": np.array([6]),
           "rest": np.array([7]), "kc_lobe": np.array(["gamma", "ab"], dtype=object)}
    kc_pn = (np.array([0, 1, 1]), np.array([2, 2, 3]))
    w = mb_probe.summarize(g, np.array([10]), np.array([4]), 0.1, 100.0, idx, kc_pn)
    assert w["kc_lobe_fractions"] == {"ab": 0.0, "gamma": 1.0}
    cfg = {**CFG, "stimuli": {"pos": ["ORN_A"], "neg": ["ORN_B"]}, "target_stimulus": "pos",
           "negative_stimulus": "neg", "loss_window_ms": 50, "orn_rate_hz": 150.0,
           "kc_min_type_size": 1, "apl_type": "APL", "windows_ms": [50],
           "lobes": {"gamma": ["KCg"], "ab": ["KCab"]}}
    m = mb_probe.measure_kc(g, LIFParams(w_scale=1.0), None, 2, 0, "cpu", cfg)
    assert 0.0 <= m["kc_lobe_fraction_min"] <= m["kc_lobe_fraction_max"] <= 1.0

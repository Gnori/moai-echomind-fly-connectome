"""F-007 준비 — 구획 표 · 3요소 규칙(짝 지은 KC 만 · 바닥) · 엔진 연결 세기(작은 그래프)."""

from __future__ import annotations

import numpy as np
import torch

from moai_echomind_fly import config
from moai_echomind_fly import plasticity as pl
from moai_echomind_fly.graph import Graph
from moai_echomind_fly.sim import Engine, LIFParams


def mb_graph() -> Graph:
    """KC 0~3 · MBON 4(MBON01) · 5(MBON11) · DAN 6(PAM01) · 7(PPL101) · 8(다른 뉴런)."""
    edges = {0: [4, 5], 1: [4, 5], 2: [4], 3: [5], 6: [4], 7: [5], 8: [0]}
    count = {(6, 4): 30, (7, 5): 30}
    src, dst, cnt = [], [], []
    for s in range(9):
        for d in edges.get(s, []):
            src.append(s)
            dst.append(d)
            cnt.append(count.get((s, d), 5))
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=9))]).astype(np.int64)
    cls = ["Kenyon_Cell"] * 4 + ["MBON", "MBON", "DAN", "DAN", "other"]
    typ = ["KCg"] * 4 + ["MBON01", "MBON11", "PAM01", "PPL101", "X"]
    cols = {"class": np.array(cls, dtype=object), "type": np.array(typ, dtype=object)}
    return Graph("mb", cols, np.arange(9), np.ones(9, np.int8), row_ptr,
                 np.array(dst, np.int32), np.array(cnt, np.int32), {})


CFG = config.load("plasticity")


def test_compartments_follow_strongest_dan_and_check_table():
    c = pl.compartments(mb_graph(), CFG)
    assert c["table"] == {"MBON01": "PAM01", "MBON11": "PPL101"}
    assert c["check"]["PAM01→MBON01"] and c["check"]["PPL101→MBON11"]


def test_pairing_weakens_only_paired_kc_in_the_dan_compartment():
    g = mb_graph()
    tf = pl.ThreeFactor(g, CFG, eta=0.05)
    assert len(tf.edges) == 6
    for _ in range(10):                                      # KC 0 · 2 와 PAM01 이 함께 발화
        tf.update(np.zeros(4, np.int32), np.array([0, 2, 6, 6], np.int32), 100.0)
    src = g.src()[tf.edges]
    dst = g.dst[tf.edges]
    weak = (np.isin(src, [0, 2])) & (dst == 4)
    assert (tf.m[weak] < 1.0).all() and (tf.m[~weak] == 1.0).all()
    tf.eta = 1e6
    tf.update(np.zeros(2, np.int32), np.array([0, 6], np.int32), 100.0)
    assert tf.m.min() == CFG["rule"]["m_min"]                # 바닥


def test_no_change_without_dan_or_with_zero_eta():
    g = mb_graph()
    tf = pl.ThreeFactor(g, CFG, eta=0.05)
    tf.update(np.zeros(3, np.int32), np.array([0, 1, 2], np.int32), 100.0)   # DAN 없음
    assert (tf.m == 1.0).all()
    z = pl.ThreeFactor(g, CFG, eta=0.0)
    z.update(np.zeros(2, np.int32), np.array([0, 6], np.int32), 100.0)
    assert (z.m == 1.0).all()


def test_apply_scales_engine_weights_on_kc_mbon_edges_only():
    g = mb_graph()
    eng = Engine(g, LIFParams(w_scale=1.0), dtype=torch.float64)
    w_before = eng.w.clone()
    tf = pl.ThreeFactor(g, CFG, eta=0.05)
    tf.m[:] = 0.5
    tf.apply(eng)
    mask = np.zeros(len(g.dst), bool)
    mask[tf.edges] = True
    assert torch.allclose(eng.w[mask], 0.5 * w_before[mask])
    assert torch.equal(eng.w[~mask], w_before[~mask])
    tf.m[:] = 1.0
    tf.apply(eng)                                            # 처음 세기 기준(누적되지 않음)
    assert torch.equal(eng.w, w_before)


def test_controls_helpers_and_verdict():
    t = {"MBON01": "PAM01", "MBON11": "PPL101", "MBON05": "PAM08"}
    s = pl.shuffled_table(t, 3)
    assert sorted(s) == sorted(t) and sorted(s.values()) == sorted(t.values())
    rows = [{"k": [10, 4]}, {"k": [10, 6]}]
    assert pl.change(rows, "k") == {"before": 20, "after": 10, "change": -0.5, "seeds_down": 2}
    pc = CFG["probe_controls"]

    def ch(x):
        return {"change": x}

    summ = {"paired": {"cs_plus.reward": ch(-0.6), "cs_minus.reward": ch(0.1)},
            "unpaired": {"cs_plus.reward": ch(-0.1)}, "shuffled": {"cs_plus.reward": ch(-0.4)},
            "off": {"cs_plus.reward": ch(0.1)},
            "punish": {"cs_plus.reward": ch(-0.05), "cs_plus.punish": ch(-0.7)}}
    v = pl.verdict_controls(summ, pc)
    assert v == {"paired": True, "cs_minus": True, "unpaired": True, "shuffled": False,
                 "off": True, "punish": True}


def test_run_condition_paired_weakens_and_off_does_not():
    g = mb_graph()
    cfg = {**CFG, "probe_controls": {**CFG["probe_controls"], "eta": 0.5, "test_ms": 20,
                                     "update_ms": 10, "unpaired_gap_s": 0.01},
           "eta": {**CFG["eta"], "pairing": {"odor_s": 0.05, "dan_s": 0.05}}}
    sets = {"table": {"MBON01": "PAM01", "MBON11": "PPL101"}, "mbon": np.array([4, 5]),
            "reward_mbon": np.array([4]), "punish_mbon": np.array([5]),
            "reward_dan": np.array([6]), "punish_dan": np.array([7]),
            "cs_plus": np.array([0, 2]), "cs_minus": np.array([3])}
    lif = LIFParams(w_scale=1.0)
    on = pl.run_condition(g, lif, cfg, None, "paired", 0, "cpu", sets)
    off = pl.run_condition(g, lif, cfg, None, "off", 0, "cpu", sets)
    assert on["changed"] > 0 and off["changed"] == 0
    for cond in ("unpaired", "punish", "shuffled"):
        r = pl.run_condition(g, lif, cfg, None, cond, 0, "cpu", sets)
        assert len(r["cs_plus.reward"]) == 2


def test_condition_names_are_strings():
    assert all(isinstance(c, str) for c in CFG["probe_controls"]["conditions"])
    assert "off" in CFG["probe_controls"]["conditions"]


def test_cs_minus_ok_uses_same_threshold():
    pc = CFG["probe_controls"]
    assert pl.cs_minus_ok({"paired": {"cs_minus.reward": {"change": -0.1}}}, pc) is True
    assert pl.cs_minus_ok({"paired": {"cs_minus.reward": {"change": -0.4}}}, pc) is False
    assert pl.cs_minus_ok({"paired": {"cs_minus.reward": {"change": None}}}, pc) is None


def test_drive_split_counts_shared_kc_and_predicts_remaining():
    g = mb_graph()
    tf = pl.ThreeFactor(g, CFG, eta=0.05, table={"MBON01": "PAM01", "MBON11": "PPL101"})
    tf.w0 = np.ones(len(tf.edges))
    src, dst = g.src()[tf.edges], g.dst[tf.edges]
    tf.m[(dst == 4) & (src == 0)] = 0.5                      # KC 0 → 보상 MBON 만 약해짐
    plus = np.array([2, 0, 0, 0])                             # CS+ = KC 0
    minus = np.array([2, 0, 2, 0])                            # CS− = KC 0 · 2
    paired = np.array([True, False, False, False])
    sp = pl.drive_split(tf, g, np.array([4]), plus, minus, paired)
    assert sp["kc_to_reward"] == 3                           # KC 0 · 1 · 2 → MBON 4
    assert sp["jaccard_all"] == 0.5 and sp["jaccard_reward"] == 0.5
    assert sp["cs_minus"]["share_with_cs_plus"] == 0.5
    assert sp["cs_minus"]["share_paired_kc"] == 0.5
    assert sp["cs_minus"]["predicted_remaining"] == 0.75
    assert sp["cs_plus"]["predicted_remaining"] == 0.5


def test_run_condition_detail_adds_split():
    g = mb_graph()
    cfg = {**CFG, "probe_controls": {**CFG["probe_controls"], "eta": 0.5, "test_ms": 20,
                                     "update_ms": 10},
           "eta": {**CFG["eta"], "pairing": {"odor_s": 0.05, "dan_s": 0.05}}}
    sets = {"table": {"MBON01": "PAM01", "MBON11": "PPL101"}, "mbon": np.array([4, 5]),
            "reward_mbon": np.array([4]), "punish_mbon": np.array([5]),
            "reward_dan": np.array([6]), "punish_dan": np.array([7]),
            "cs_plus": np.array([0, 2]), "cs_minus": np.array([3])}
    r = pl.run_condition(g, LIFParams(w_scale=1.0), cfg, None, "paired", 0, "cpu", sets,
                         detail=True)
    assert {"jaccard_all", "jaccard_reward", "cs_plus", "cs_minus"} <= set(r["split"])


def test_rewire_subset_keeps_other_edges_and_degrees():
    from moai_echomind_fly import graph as gm

    g = mb_graph()
    r = gm.rewire_subset(g, 3, {"class": ["Kenyon_Cell"]}, {"class": ["MBON"]}, "kcmbon")
    src = g.src()
    m = np.isin(g.columns["class"][src], ["Kenyon_Cell"]) & np.isin(g.columns["class"][g.dst],
                                                                    ["MBON"])
    assert np.array_equal(r.dst[~m], g.dst[~m])                     # 갈래 밖은 그대로
    assert sorted(r.dst[m].tolist()) == sorted(g.dst[m].tolist())   # 도착 연결 수 그대로
    assert np.isin(r.dst[m], [4, 5]).all()                          # 갈래 안에서만
    assert gm.brain_graph(g, "real") is g
    assert gm.brain_graph(g, "kcmbon-s3").meta["rewire_edges"] == int(m.sum())

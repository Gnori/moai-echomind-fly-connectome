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

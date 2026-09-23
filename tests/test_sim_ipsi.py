"""sim-v2 같은 쪽 방출 비대칭 — 규칙이 걸리는 연결만 바뀌나(작은 가짜 그래프 · 가벼운 시험)."""

from __future__ import annotations

import numpy as np

from moai_echomind_fly.graph import Graph
from moai_echomind_fly.sim import ipsi_scale


def _graph() -> Graph:
    # 0 = 왼쪽 후각 수용 뉴런(rootSide L) · 1 = 왼쪽 투사 뉴런 · 2 = 오른쪽 투사 뉴런
    # 3 = 왼쪽 다른 뉴런
    cols = {
        "class": np.array(["olfactory", "ALPN", "ALPN", "LHLN"], dtype=object),
        "type": np.array(["ORN_DM1", "DM1_lPN", "DM1_lPN", "X"], dtype=object),
        "somaSide": np.array(["", "L", "R", "L"], dtype=object),
        "rootSide": np.array(["L", "", "", ""], dtype=object),
        "instance": np.array(["a", "b_L", "c_R", "d_L"], dtype=object),
    }
    # 연결: 0→1(같은 쪽) · 0→2(반대쪽) · 3→1(후각 아님 · 같은 쪽)
    row_ptr = np.array([0, 2, 2, 2, 3])
    dst = np.array([1, 2, 1], dtype=np.int32)
    count = np.array([5, 5, 5], dtype=np.int32)
    return Graph("t", cols, np.arange(4), np.ones(4, np.int8), row_ptr, dst, count, {})


def test_ipsi_scale_only_same_side_edges_from_rule_class():
    s = ipsi_scale(_graph(), {"class": ["olfactory"], "ipsi_factor": 1.4})
    assert s.tolist() == [1.4, 1.0, 1.0]

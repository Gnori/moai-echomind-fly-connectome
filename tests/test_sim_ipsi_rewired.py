"""sim-v2 같은 쪽 방출 규칙이 **섞은 배선**에도 같은 뜻으로 걸리나.

출발이 olfactory 이고 (섞인) 도착이 같은 쪽인 연결만 × 1.4 여야 한다(가벼운 시험).
"""

from __future__ import annotations

import numpy as np

from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.graph import Graph
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import ipsi_scale

RULE = {"class": ["olfactory"], "ipsi_factor": 1.4}


def _graph() -> Graph:
    # 0 = 왼쪽 ORN · 1 = 오른쪽 ORN · 2 · 4 = 왼쪽 투사 뉴런 · 3 = 오른쪽 투사 뉴런
    # 5 = 왼쪽 다른 뉴런(후각 아님)
    cols = {
        "class": np.array(["olfactory", "olfactory", "ALPN", "ALPN", "ALPN", "LHLN"],
                          dtype=object),
        "type": np.array(["ORN_DM1", "ORN_DM1", "DM1_lPN", "DM1_lPN", "DM1_lPN", "X"],
                         dtype=object),
        "somaSide": np.array(["", "", "L", "R", "L", "L"], dtype=object),
        "rootSide": np.array(["L", "R", "", "", "", ""], dtype=object),
        "instance": np.array(["a", "b", "c_L", "d_R", "e_L", "f_L"], dtype=object),
    }
    # 0→2 · 0→3 · 1→2 · 1→3 · 5→2 · 5→4
    row_ptr = np.array([0, 2, 4, 4, 4, 4, 6], dtype=np.int64)
    dst = np.array([2, 3, 2, 3, 2, 4], dtype=np.int32)
    count = np.full(6, 5, dtype=np.int32)
    return Graph("t", cols, np.arange(6, dtype=np.int64), np.ones(6, np.int8), row_ptr, dst,
                 count, {})


def test_ipsi_scale_on_rewired_graph_marks_only_olfactory_to_same_side_edges():
    g = _graph()
    assert ipsi_scale(g, RULE).tolist() == [1.4, 1.0, 1.0, 1.4, 1.0, 1.0]
    side = side_of(g)
    seen_change = False
    for seed in range(6):
        r = graph_mod.rewire(g, seed)
        assert r.row_ptr.tolist() == g.row_ptr.tolist() and r.count.tolist() == g.count.tolist()
        src = r.src()
        expect = np.where((r.columns["class"][src] == "olfactory") & np.isin(side[src], ["L", "R"])
                          & (side[src] == side[r.dst]), 1.4, 1.0)
        got = ipsi_scale(r, RULE)
        assert got.tolist() == expect.tolist()
        assert set(got[src >= 2].tolist()) == {1.0}          # 후각이 아닌 출발은 늘 1
        seen_change |= r.dst.tolist() != g.dst.tolist()
    assert seen_change                                         # 적어도 한 씨앗은 실제로 섞였다

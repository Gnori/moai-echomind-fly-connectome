"""retino-v1 — 시야 고도 계산(R-7469 F-008 2단계)을 작은 가짜 그래프로 확인한다.

가벼운 시험 · 실제 파일 없음. (a) 좌표 있는 입력의 시냅스 가중 평균 (b) 좌표 없는 입력만 받으면
NaN · 가중 0 (c) `build()` 에서 입력 시냅스가 `min_input_synapses` 미만이면 missing.
raw feather 는 `column_hex` 를 바꿔 끼운다.
"""

from __future__ import annotations

import numpy as np

from moai_echomind_fly import retino
from moai_echomind_fly.graph import Graph

# 0 · 1 = 좌표 있는 기둥 뉴런(Mi1) · 2 = T4a(좌표 없음 · 두 단계) · 3 = LC4(0 · 1 에서 받음)
# 4 = LC6(좌표 없는 5 에서만 받음) · 5 = 다른 뉴런 · 6 = LC17(T4a 에서 5 시냅스만)
_TYPES = ["Mi1", "Mi1", "T4a", "LC4", "LC6", "LoLP1", "LC17"]
_SUPER = ["ol_intrinsic", "ol_intrinsic", "ol_intrinsic", "visual_projection",
          "visual_projection", "ol_intrinsic", "visual_projection"]


def _graph() -> Graph:
    n = len(_TYPES)
    cols = {
        "type": np.array(_TYPES, dtype=object),
        "superclass": np.array(_SUPER, dtype=object),
        "class": np.array([""] * n, dtype=object),
        "somaSide": np.array(["R", "R", "R", "R", "L", "R", "L"], dtype=object),
        "rootSide": np.array([""] * n, dtype=object),
        "instance": np.array([f"{t}_R" for t in _TYPES], dtype=object),
    }
    # 연결(출발 → 도착 · 시냅스 수): 0→2 10 · 0→3 30 · 1→3 10 · 2→6 5 · 5→4 50
    row_ptr = np.array([0, 2, 3, 4, 4, 4, 5, 5], dtype=np.int64)
    dst = np.array([2, 3, 3, 6, 4], dtype=np.int32)
    count = np.array([10, 30, 10, 5, 50], dtype=np.int32)
    return Graph("toy", cols, np.arange(n, dtype=np.int64), np.ones(n, np.int8), row_ptr, dst,
                 count, {})


def _hexes(n: int) -> np.ndarray:
    h = np.full((n, 2), np.nan)
    h[0] = (10.0, 10.0)      # y = 20
    h[1] = (30.0, 40.0)      # y = 70
    return h


def test_weighted_input_hex_is_synapse_weighted_mean():
    g = _graph()
    hx, w = retino.weighted_input_hex(g, np.array([3]), _hexes(g.n))
    assert w.tolist() == [40.0]
    np.testing.assert_allclose(hx[0], [(30 * 10 + 10 * 30) / 40, (30 * 10 + 10 * 40) / 40])


def test_weighted_input_hex_without_coordinate_inputs_is_nan_and_zero_weight():
    g = _graph()
    hx, w = retino.weighted_input_hex(g, np.array([4]), _hexes(g.n))
    assert w.tolist() == [0.0]
    assert np.isnan(hx[0]).all()


def test_build_marks_low_synapse_and_no_coordinate_neurons_missing(monkeypatch):
    g = _graph()
    monkeypatch.setattr(retino, "column_hex", lambda graph, raw=None: _hexes(graph.n))
    cfg = {"version": "retino-test", "select": {"superclass": ["visual_projection"]},
           "column_hex_types": "auto", "two_hop_types_prefix": ["T4", "T5"],
           "elevation_range_deg": [-70.0, 90.0], "min_input_synapses": 20}
    res = retino.build(g, cfg)
    assert res["rule"] == "retino-test" and res["y_range"] == [20.0, 70.0]
    got = {n["type"]: n for n in res["neurons"]}
    assert list(got) == ["LC4"]                           # LC6(좌표 없음) · LC17(5 < 20) 은 빠진다
    assert got["LC4"]["input_synapses"] == 40 and got["LC4"]["side"] == "R"
    # 가중 평균 y = 32.5 → (32.5 − 20) / 50 × 160 − 70 = −30
    assert got["LC4"]["elevation_deg"] == -30.0
    assert res["missing"] == {"LC6": 1, "LC17": 1}

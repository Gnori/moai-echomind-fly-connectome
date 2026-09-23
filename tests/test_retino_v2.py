"""retino-v2 — 방위(앞 · 뒤 축)와 뒤섞은 대응표를 작은 가짜 그래프로 확인한다.

R-7469 F-008 · I v2 §9.2. 가벼운 시험 · 실제 파일 없음(raw feather 는 `column_hex` 를 바꿔 끼운다).
(a) 오른눈: 앞쪽(h = hex1 − hex2 큰) 입력을 받는 뉴런의 방위가 0° 에 가깝고 뒤쪽은 멀다
(b) 왼눈은 같은 기둥에서 부호가 반대 (c) 고도는 v1 과 같은 식
(d) 옛 판 설정(방위 범위 없음)이면 방위 칸이 없다
(e) `shuffled()` 는 같은 종류 · 같은 눈 안에서만 값을 바꾸고 다른 칸은 그대로다.
"""

from __future__ import annotations

import numpy as np
import pytest

from moai_echomind_fly import retino
from moai_echomind_fly.graph import Graph

# 0~3 = 좌표 있는 기둥 뉴런(Mi1) · 4~9 = 시각 투사 뉴런
#   0: (30, 10) h = +20(앞) v = 40 · 1: (10, 30) h = −20(뒤) v = 40
#   2: (20, 20) h = 0 v = 40 · 3: (35, 35) h = 0 v = 70(위)
#   4 LC4_R ← 0(앞) · 5 LC4_R ← 1(뒤) · 6 LC4_L ← 0(앞) · 7 LC4_L ← 1(뒤)
#   8 LPLC2_R ← 2(가운데) · 9 LC4_R ← 3(위)
_TYPES = ["Mi1", "Mi1", "Mi1", "Mi1", "LC4", "LC4", "LC4", "LC4", "LPLC2", "LC4"]
_SIDES = ["R", "R", "R", "R", "R", "R", "L", "L", "R", "R"]


def _graph() -> Graph:
    n = len(_TYPES)
    cols = {
        "type": np.array(_TYPES, dtype=object),
        "superclass": np.array(["ol_intrinsic"] * 4 + ["visual_projection"] * 6, dtype=object),
        "class": np.array([""] * n, dtype=object),
        "somaSide": np.array(_SIDES, dtype=object),
        "rootSide": np.array([""] * n, dtype=object),
        "instance": np.array([f"{t}_{s}" for t, s in zip(_TYPES, _SIDES)], dtype=object),
    }
    # 연결(출발 → 도착 · 시냅스 수): 0→4 30 · 0→6 30 · 1→5 30 · 1→7 30 · 2→8 30 · 3→9 30
    row_ptr = np.array([0, 2, 4, 5, 6, 6, 6, 6, 6, 6, 6], dtype=np.int64)
    dst = np.array([4, 6, 5, 7, 8, 9], dtype=np.int32)
    count = np.array([30, 30, 30, 30, 30, 30], dtype=np.int32)
    return Graph("toy", cols, np.arange(n, dtype=np.int64), np.ones(n, np.int8), row_ptr, dst,
                 count, {})


def _hexes(n: int) -> np.ndarray:
    h = np.full((n, 2), np.nan)
    h[0] = (30.0, 10.0)
    h[1] = (10.0, 30.0)
    h[2] = (20.0, 20.0)
    h[3] = (35.0, 35.0)
    return h


_CFG = {"version": "retino-v2-test", "select": {"superclass": ["visual_projection"]},
        "column_hex_types": "auto", "two_hop_types_prefix": ["T4", "T5"],
        "elevation_range_deg": [-70.0, 90.0], "azimuth_range_deg": [-170.0, 10.0],
        "min_input_synapses": 20}


@pytest.fixture
def res(monkeypatch):
    monkeypatch.setattr(retino, "column_hex", lambda graph, raw=None: _hexes(graph.n))
    return retino.build(_graph(), dict(_CFG))


def _by_idx(res):
    return {n["idx"]: n for n in res["neurons"]}


def test_axes_and_ranges(res):
    assert res["rule"] == "retino-v2-test"
    assert res["h_range"] == [-20.0, 20.0] and res["v_range"] == [40.0, 70.0]
    assert res["azimuth_range_deg"] == [-170.0, 10.0]
    got = _by_idx(res)
    assert got[4]["h"] == 20.0 and got[5]["h"] == -20.0 and got[4]["v"] == 40.0


def test_right_eye_front_is_near_zero_and_back_is_far(res):
    got = _by_idx(res)
    assert got[4]["azimuth_deg"] == 10.0          # 앞(h 최대) → 범위 위 = 정중선 너머 10°
    assert got[5]["azimuth_deg"] == -170.0        # 뒤(h 최소) → 범위 아래
    assert abs(got[4]["azimuth_deg"]) < abs(got[5]["azimuth_deg"])
    assert got[8]["azimuth_deg"] == -80.0         # h = 0 → 가운데
    assert got[4]["azimuth_deg"] > got[5]["azimuth_deg"]   # 오른눈: h 클수록 방위가 크다(+ 왼쪽)


def test_left_eye_is_mirror_of_right(res):
    got = _by_idx(res)
    assert got[6]["azimuth_deg"] == -got[4]["azimuth_deg"] == -10.0
    assert got[7]["azimuth_deg"] == -got[5]["azimuth_deg"] == 170.0


def test_elevation_is_v1_formula(res):
    got = _by_idx(res)
    assert got[4]["elevation_deg"] == -70.0       # v 최소
    assert got[9]["elevation_deg"] == 90.0        # v 최대
    assert got[9]["azimuth_deg"] == -80.0         # 위쪽 기둥의 h = 0


def test_old_config_without_azimuth_range_has_no_azimuth(monkeypatch):
    monkeypatch.setattr(retino, "column_hex", lambda graph, raw=None: _hexes(graph.n))
    cfg = {k: v for k, v in _CFG.items() if k != "azimuth_range_deg"}
    cfg["version"] = "retino-v1-test"
    r = retino.build(_graph(), cfg)
    assert all("azimuth_deg" not in n for n in r["neurons"])
    assert "azimuth_range_deg" not in r and r["azimuth"].startswith("없음")
    assert _by_idx(r)[4]["elevation_deg"] == -70.0
    with pytest.raises(ValueError):
        retino.shuffled(r, "azimuth", 0)


def test_shuffled_permutes_only_within_type_and_side(res):
    seeds_changed = 0
    for seed in range(6):
        sh = retino.shuffled(res, "azimuth", seed)
        assert sh["rule"] == "retino-v2-test.shuffled-azimuth-s" + str(seed)
        assert sh["base_rule"] == "retino-v2-test"
        assert sh["shuffle"] == {"what": "azimuth", "key": "azimuth_deg", "seed": seed,
                                 "within": ["type", "side"]}
        a, b = _by_idx(res), _by_idx(sh)
        for i in a:                                         # 다른 칸은 그대로
            for k in ("type", "side", "hex", "h", "v", "elevation_deg", "input_synapses"):
                assert a[i][k] == b[i][k]
        for key in {(n["type"], n["side"]) for n in res["neurons"]}:   # 종류 × 눈 안 같은 값 집합
            va = sorted(n["azimuth_deg"] for n in res["neurons"]
                        if (n["type"], n["side"]) == key)
            vb = sorted(n["azimuth_deg"] for n in sh["neurons"]
                        if (n["type"], n["side"]) == key)
            assert va == vb
        assert b[6]["azimuth_deg"] in (-10.0, 170.0) and b[8]["azimuth_deg"] == -80.0
        if any(a[i]["azimuth_deg"] != b[i]["azimuth_deg"] for i in a):
            seeds_changed += 1
    assert seeds_changed > 0                                 # 어떤 씨앗에서는 실제로 섞인다
    assert res["rule"] == "retino-v2-test"                   # 원본은 안 바뀐다


def test_shuffled_elevation_leaves_azimuth(res):
    sh = retino.shuffled(res, "elevation", 1)
    a, b = _by_idx(res), _by_idx(sh)
    assert all(a[i]["azimuth_deg"] == b[i]["azimuth_deg"] for i in a)
    lc4_r = [i for i in a if a[i]["type"] == "LC4" and a[i]["side"] == "R"]
    before = sorted(a[i]["elevation_deg"] for i in lc4_r)
    assert before == sorted(b[i]["elevation_deg"] for i in lc4_r)
    with pytest.raises(ValueError):
        retino.shuffled(res, "colour", 0)


def test_summarize_reports_azimuth_per_eye(res):
    lines = retino.summarize(res, ("LC4", "LPLC2"))
    assert len(lines) == 2 and lines[0].startswith("  LC4") and "방위 R(3)" in lines[0]
    assert "방위 L(2)" in lines[0] and "방위 R(1)" in lines[1]

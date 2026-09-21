"""F-002: 공식 파일로 컴파일한 그래프가 08 실측과 맞는다."""

from __future__ import annotations

import numpy as np
import pytest

from moai_echomind_fly import graph

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def g() -> graph.Graph:
    return graph.load()


def test_counts_match_research_note_08(g):
    assert g.n == 165_122
    assert g.e == 25_563_197  # Fly.exe · ConnecTorch 가 적은 Traced 연결 수
    nt = g.meta["nt_counts"]
    assert nt["acetylcholine"] == 103_718
    assert nt["glutamate"] + nt["gaba"] + nt["histamine"] == 57_261
    assert nt["dopamine"] + nt["serotonin"] + nt["octopamine"] == 541
    assert nt["unclear"] == 3_100
    assert nt["missing"] == 502  # 신경전달물질 행이 없는 Traced 뉴런


def test_kenyon_cells_are_excitatory(g):
    """predicted_nt 가 아니라 consensus_nt 를 썼다는 증거 — 케니언 세포 4,064 개 전부 +."""
    kc = g.select(klass="Kenyon_Cell")
    assert len(kc) == 4_064
    assert (g.sign[kc] == 1).all()


def test_weight_is_sign_times_count_csr_sorted(g):
    assert g.row_ptr[0] == 0 and g.row_ptr[-1] == g.e
    assert (np.diff(g.row_ptr) >= 0).all()
    assert (g.count >= 1).all()


def test_rewired_preserves_in_and_out_degree(g):
    r = graph.rewire(g, seed=0)
    assert np.array_equal(np.diff(r.row_ptr), np.diff(g.row_ptr))
    assert np.array_equal(np.bincount(r.dst, minlength=g.n), np.bincount(g.dst, minlength=g.n))
    assert not np.array_equal(r.dst, g.dst)
    assert np.array_equal(r.count, g.count)

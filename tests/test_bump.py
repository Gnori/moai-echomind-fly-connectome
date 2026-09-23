"""나침반 봉우리 검사(R-7469 F-008 3단계) 부품 — EPG 단위 고르기 · 위상 · 원형 상관(가벼운 시험)."""

from __future__ import annotations

import numpy as np

from moai_echomind_fly.loop import bump


def test_epg_units_drop_glomerulus_9_and_non_epg_units():
    units = ["EPG(PB08)_R1:R", "EPG(PB08)_R9:R", "EPG(PB08)_L4:L",
             "PEN_a(PEN1)(PB05)_L3:L", "EPGt(PB01)_R2:R", "Delta7(PB15)_L2R7:M"]
    got = bump.epg_units(units)
    assert [i for i, _ in got] == [0, 2]
    np.testing.assert_allclose([a for _, a in got], np.deg2rad([0.0, 135.0]))


def test_phase_of_gives_zero_length_when_nothing_fires():
    units = [f"EPG(PB08)_R{i}:R" for i in range(1, 9)] + ["PFNd(PB04)_L2_C1:L"]
    x = np.zeros((5, 2 * 3 * len(units)), np.float32)
    phase, length = bump.phase_of(x, units)
    assert phase.shape == (5,) and length.tolist() == [0.0] * 5


def test_circ_corr_is_one_for_identical_angles_and_small_for_independent_ones():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 2000)
    assert abs(bump.circ_corr(a, a) - 1.0) < 1e-9
    b = rng.uniform(0, 2 * np.pi, 2000)
    assert abs(bump.circ_corr(a, b)) < 0.1

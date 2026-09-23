"""판정 통계 stats-v1 — 에피소드 부호 뒤집기 · Holm · 부트스트랩 · 판정(가벼운 시험)."""

from __future__ import annotations

import numpy as np

from moai_echomind_fly.loop import stats


def test_sign_flip_exact_counts_all_cases():
    # 에피소드 3 개 모두 +1 → 2^3 경우 중 합 ≥ 3 은 하나 → p = 1/8
    assert stats.sign_flip_p(np.array([1.0, 1.0, 1.0])) == 1 / 8
    assert stats.sign_flip_p(np.array([0.0, 0.0])) == 1.0          # 차이 없음 → 1


def test_sign_flip_monte_carlo_when_many_episodes():
    rule = {**stats.STATS_RULE, "exact_max_episodes": 4, "permutations": 2000}
    strong = stats.sign_flip_p(np.ones(30), rule)
    noise = stats.sign_flip_p(np.random.default_rng(1).standard_normal(30), rule)
    assert strong < 0.01 and noise > 0.05


def test_holm_is_monotone_and_capped():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06   # 누적 최대
    assert all(v <= 1.0 for v in stats.holm({"x": 0.9, "y": 0.8}).values())


def test_judge_cell_passes_only_when_real_beats_ensemble_blockwise():
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(12), 9)                                # 12 에피소드 × 9 결정
    base = np.zeros(len(ep))
    real = rng.random(len(ep)) < 0.85
    rewired = [rng.random(len(ep)) < 0.5 for _ in range(5)]
    cells = {"vision|x": stats.judge_cell(real, rewired, base, ep),
             "noise|x": stats.judge_cell(rewired[0], rewired[1:], base, ep)}
    for c in cells.values():
        c["judge"] = True
    stats.finalize(cells)
    assert cells["vision|x"]["pass"] is True
    assert cells["noise|x"]["pass"] is False
    lo, hi = cells["vision|x"]["ci95_real"]
    assert lo < cells["vision|x"]["real"] < hi


def test_excluded_cells_are_not_corrected_or_judged():
    ep = np.repeat(np.arange(4), 3)
    c = stats.judge_cell(np.ones(12, bool), [np.zeros(12, bool)], np.zeros(12), ep)
    cells = {"motor|x": {**c, "judge": False}}
    stats.finalize(cells)
    assert cells["motor|x"]["pass"] is None and cells["motor|x"]["p_holm"] is None

"""냄새 여러 창 읽기(odor-windows-v1 · 노트 14 M2) — 창 평균은 지난 결정만 · 에피소드 안에서만."""

from __future__ import annotations

import numpy as np

from moai_echomind_fly.loop.odor_windows import delta_features, icc, window_mean


def _rows(seeds_decisions, headings):
    return [{"seed": s, "decision": d, "truth": {"heading_deg": h}}
            for (s, d), h in zip(seeds_decisions, headings)]


def test_window_mean_uses_only_past_decisions_within_episode():
    # 행 순서를 섞어 둔다 — 결정 번호로 정렬해야 한다
    rows = _rows([(1, 1), (1, 0), (1, 2), (2, 0), (2, 1)], [0] * 5)
    x = np.array([[2.0], [0.0], [4.0], [100.0], [200.0]])
    w = window_mean(x, rows, 2)
    assert w[:, 0].tolist() == [1.0, 0.0, 3.0, 100.0, 150.0]   # 에피소드 2 는 1 의 값을 안 섞음


def test_delta_features_first_decision_zero_and_heading_wraps():
    rows = _rows([(1, 0), (1, 1)], [170.0, -170.0])            # 실제 회전 +20°
    x = np.array([[1.0, 1.0], [1.5, 1.5]])                      # 세기 +50 %
    d = delta_features(x, rows)
    assert d[0].tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(d[1], [0.5, 20 / 90, 0.5 * 20 / 90])


def test_icc_is_one_when_episodes_are_all_right_or_all_wrong():
    ep = np.repeat([1, 2, 3, 4], 5)
    assert icc(np.repeat([1, 0, 1, 0], 5).astype(bool), ep) > 0.99
    rng = np.random.default_rng(0)
    assert abs(icc(rng.random(2000) < 0.5, np.repeat(np.arange(400), 5))) < 0.1

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


def test_compress_fit_picks_k_by_variance_rule():
    from moai_echomind_fly.loop.compress import apply, fit_channel

    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3)) @ rng.standard_normal((3, 100))   # 계수 3 인 자료
    m = fit_channel(x, {"var_target": 0.95, "k_min": 2, "k_max": 10, "center": True})
    assert m["components"].shape == (3, 100) and m["explained"] >= 0.95
    z = apply({"c": m}, {"c": x.astype(np.float32)})["c"]
    assert z.shape == (200, 3)
    m2 = fit_channel(x, {"var_target": 0.95, "k_min": 5, "k_max": 10, "center": True})
    assert m2["components"].shape[0] == 5                              # 아래 한계


def test_bump_phase_follows_planted_heading():
    from moai_echomind_fly.loop import bump

    units = [f"EPG(PB08)_R{i}:R" for i in range(1, 9)] + ["PFNd(PB04)_L2_C1:L"]
    heading = np.deg2rad(np.arange(0, 360, 15.0))
    ang = np.deg2rad(np.arange(8) * 45.0)
    rate = np.clip(np.cos(ang[None, :] - heading[:, None]), 0, None) * 20   # 방향을 따라가는 봉우리
    x = np.concatenate([np.tile(np.c_[rate, np.zeros(len(rate))], 3),
                        np.zeros((len(rate), 3 * 9))], axis=1)
    phase, length = bump.phase_of(x, units)
    assert abs(bump.circ_corr(phase, heading)) > 0.95 and np.median(length) > 0.3


def test_value_spans_and_token_weights_mark_only_field_values():
    from moai_echomind_fly.loop.train_cli import token_weights, value_spans

    cap = "다가오는 물체: 왼쪽 · 중간 · 다가옴 / 먹이 냄새: 없음 / 다리 접촉: 6개 / 소리: 없음"
    spans = value_spans(cap, "cap")
    assert [cap[a:b] for a, b in spans] == ["왼쪽 · 중간 · 다가옴", "없음", "6개", "없음"]
    act = '{"walk":0.8,"turn":"none","turn_strength":0.0,"stop":false,"groom":false}'
    assert [act[a:b] for a, b in value_spans(act, "act")] == [
        "0.8", '"none"', "0.0", "false", "false"]
    # 토큰 가상 구간: "다가오는 물체: "(틀) · "왼쪽"(값)
    w = token_weights([(0, 8), (9, 11)], spans, 0.1)
    assert w == [0.1, 1.0]


def test_scorecard_judge_needs_to_beat_every_rewired_and_be_significant():
    from moai_echomind_fly.loop.scorecard import judge

    def eps(hits):
        return [{"seed": s, "reached": bool(h), "looming": False, "looming_hit": None,
                 "reach_ms": 1.0 if h else None, "final_food_mm": 1.0, "decisions": 1}
                for s, h in enumerate(hits)]

    good = eps([1] * 40)
    bad = [eps([1] * 5 + [0] * 35) for _ in range(3)]
    v = judge({"vector": good, "rewired-s0": bad[0], "rewired-s1": bad[1], "rewired-s2": bad[2]})
    assert v["vector_beats_rewired"] is True and v["p_sign_flip"] < 0.05
    tie = judge({"vector": bad[0], "rewired-s0": good, "rewired-s1": bad[1], "rewired-s2": bad[2]})
    assert tie["vector_beats_rewired"] is False

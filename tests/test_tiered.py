"""두 층 제어 · 빠른 정책 (R-1027 · P loop Task 1 · 2) — 게이트 조건 · 정직 · 회귀(always)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from moai_echomind_fly.loop import fast_policy as fp
from moai_echomind_fly.loop.tiered import Gate, TieredPolicy

TH = {"max_gap_decisions": 5, "min_interval_decisions": 0, "surprise_q": 0.9,
      "confidence_q": 0.1, "conflict_q": 0.9, "reflex_threat_min": None}
ABS = {"surprise": 1.0, "confidence": 0.3, "conflict": 0.5}     # 분위수 → 절댓값(흉내)


class StubFast:
    """z[0] = 돌기 확신도 · z[1] = 위험 점수 · z[2] 변화 = 놀람 — 게이트가 읽는 값만 흉내."""

    def predict(self, z):
        a = {"walk": 0.6, "turn": "left", "turn_strength": 0.5, "stop": False, "groom": False}
        t = {"walk": 1.0, "turn": "right", "turn_strength": 1.0, "stop": False, "groom": False}
        return fp.FastOut(a, "left", float(z[0]), float(z[1]), False, t)

    def surprise(self, z_prev, a_prev, z):
        return 0.0 if z_prev is None else float(abs(z[2] - z_prev[2]))

    def quantile(self, signal, q):
        return ABS[signal]


class StubLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, vectors, truth, obs, d):
        self.calls += 1
        return {"walk": 0.8, "turn": "none", "turn_strength": 0.0, "stop": False,
                "groom": False}, {"source": "vector"}


def _policy(mode, th=None, rate=None, **kw):
    gate = Gate({"mode": mode, "thresholds": th or TH}, StubFast(),
                sniff_decisions=kw.pop("sniff", 0), rate=rate)
    llm = StubLLM() if mode != "never" else None
    return TieredPolicy(StubFast(), llm, gate, lambda v: np.asarray(v, float), **kw), llm


CALM = [0.9, 0.0, 0.0]


def test_always_calls_llm_every_decision_like_today():
    pol, llm = _policy("always")
    tiers = [pol(CALM, None, None, d)[1]["tier"] for d in range(6)]
    assert tiers == ["llm"] * 6 and llm.calls == 6


def test_never_is_fast_only():
    pol, _ = _policy("never")
    out = [pol(CALM, None, None, d) for d in range(4)]
    assert all(i["tier"] == "fast" for _, i in out) and out[0][0]["turn"] == "left"


def test_gated_triggers_goal_gap_ambiguous_conflict_surprise():
    pol, llm = _policy("gated")
    got = [pol(CALM, None, None, 0)[1]["trigger"]]                # 첫 결정 = 목표
    got += [pol(CALM, None, None, d)[1]["trigger"] for d in (1, 2, 3, 4)]
    got.append(pol(CALM, None, None, 5)[1]["trigger"])            # 호출 뒤 5 결정 = 시간 상한
    got.append(pol([0.1, 0.0, 0.0], None, None, 6)[1]["trigger"])  # 확신도 낮음
    got.append(pol([0.9, 0.9, 0.0], None, None, 7)[1]["trigger"])  # 위험 × 먹이 쪽 0.81 > 0.5
    got.append(pol([0.9, 0.0, 5.0], None, None, 8)[1]["trigger"])  # 전방 오차 큼(놀람)
    assert got == ["goal", None, None, None, None, "gap", "ambiguous", "conflict", "surprise"]
    assert llm.calls == 5


def test_min_interval_and_sniff_suppress_calls():
    """호출 뒤 k 결정은 안 부름 · 냄새 맡기 동안은 안 부르고 끝나는 결정에서 목표."""
    pol, llm = _policy("gated", th={**TH, "min_interval_decisions": 2}, sniff=2)
    trig = [pol([0.1, 0.0, 0.0], None, None, d)[1]["trigger"] for d in range(7)]
    assert trig == [None, None, "goal", None, "ambiguous", None, "ambiguous"]
    info = pol(CALM, None, None, 7)[1]
    assert len(info["z"]) == 3 and "surprise" in info and "conflict" in info


def test_conditions_can_be_switched_off_before_results():
    gate = Gate({"mode": "gated", "thresholds": {**TH, "conflict_q": None},
                 "conditions": ["gap", "ambiguous", "surprise"]}, StubFast())
    fo = StubFast().predict([0.9, 0.9, 0.0])
    gate.reset()
    assert gate.trigger(fo, 0) == "goal"
    gate.called(0)
    assert gate.trigger(fo, 1) is None                           # 충돌 조건을 뺐다


def test_gated_refuses_unlocked_thresholds():
    with pytest.raises(ValueError, match="문턱 미정"):
        Gate({"mode": "gated", "thresholds": {**TH, "confidence_q": None}}, StubFast())
    with pytest.raises(ValueError, match="문턱 미정"):
        Gate({"mode": "gated", "thresholds": {**TH, "min_interval_decisions": None}},
             StubFast())


def test_random_and_periodic_match_a_rate():
    rnd, _ = _policy("random", rate=0.25)
    rnd.reset(seed=7)
    trig = [rnd(CALM, None, None, d)[1]["trigger"] for d in range(2001)]
    assert trig[0] == "goal" and 0.2 < trig.count("random") / 2000 < 0.3
    again, _ = _policy("random", rate=0.25)
    again.reset(seed=7)
    assert [again(CALM, None, None, d)[1]["trigger"] for d in range(2001)] == trig  # 씨앗마다 같음
    per, _ = _policy("periodic", rate=0.25)
    trig = [per(CALM, None, None, d)[1]["trigger"] for d in range(9)]
    assert trig == ["goal", None, None, None, "periodic", None, None, None, "periodic"]
    with pytest.raises(ValueError, match="rate"):
        _policy("random")


def test_truth_is_never_read_by_the_tiers():
    """참값을 망가뜨려도 층 결정이 같다(정직 · P §1.2) — LLM 은 참값을 받기만."""
    a, _ = _policy("gated")
    b, _ = _policy("gated")
    feats = [CALM, CALM, [0.1, 0.0, 0.0], [0.9, 0.9, 0.0], [0.9, 0.0, 5.0]]
    ta = [a(z, {"food": 1}, None, d)[1]["tier"] for d, z in enumerate(feats)]
    tb = [b(z, "쓰레기", object(), d)[1]["tier"] for d, z in enumerate(feats)]
    assert ta == tb


def test_hold_keeps_last_llm_action_and_reflex_skips_llm():
    pol, llm = _policy("gated", between_calls="hold")
    first = pol(CALM, None, None, 0)[0]
    assert pol(CALM, None, None, 1)[0] == first        # 부르지 않은 결정은 마지막 LLM 행동
    rx, _ = _policy("gated", reflex_threat_min=0.8)
    act, info = rx([0.9, 0.95, 0.0], None, None, 0)
    assert info["tier"] == "reflex" and act["turn_strength"] == 1.0


def test_fast_policy_recovers_planted_rule_and_round_trips(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 6))
    turn = np.where(x[:, 0] > 0.5, "left", np.where(x[:, 0] < -0.5, "right", "none")).tolist()
    stop = (x[:, 1] > 1.5).tolist()
    threat = (x[:, 2] > 1.2).tolist()
    groups = np.repeat(np.arange(60), 10)
    heads = {}
    for h, y, cl in (("turn", turn, fp.TURNS), ("stop", stop, [False, True]),
                     ("threat", threat, [False, True])):
        w, b, _ = fp._fit_head(x, y, cl, groups, fp.FAST_RULE)
        heads[h] = {"w": w, "b": b}
    pol = fp.FastPolicy(heads, ["a"], 0.5, {"rule": fp.FAST_RULE})
    acc = np.mean([pol.predict(z).turn == t for z, t in zip(x, turn)])
    assert acc > 0.8
    pol.save(tmp_path / "p")
    back = fp.FastPolicy.load(tmp_path / "p")
    assert back.predict(x[3]).turn == pol.predict(x[3]).turn
    t0 = time.perf_counter()
    for z in x[:200]:
        pol.predict(z)
    assert (time.perf_counter() - t0) / 200 < 0.005                     # 결정당 5 ms 아래


def test_labels_follow_teacher_branches():
    assert fp.labels({"walk": 1.0, "turn": "right", "turn_strength": 1.0, "stop": False}) == (
        "right", False, True)
    assert fp.labels({"walk": 0.6, "turn": "left", "turn_strength": 1.0, "stop": False}) == (
        "left", False, False)
    assert fp.labels({"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True})[1] is True


def test_scorecard_parses_tier_configs_and_keeps_old_names():
    from moai_echomind_fly.loop.scorecard import parse_config

    assert parse_config("vector") == {"base": "vector", "brain": "real", "tier_mode": None}
    assert parse_config("rewired-s1") == {"base": "rewired-s1", "brain": "rewired-s1",
                                          "tier_mode": None}
    assert parse_config("no_llm")["tier_mode"] is None
    assert parse_config("vector-gated") == {"base": "vector", "brain": "real",
                                            "tier_mode": "gated"}
    assert parse_config("rewired-s2-gated")["brain"] == "rewired-s2"
    assert parse_config("fast-only") == {"base": "vector", "brain": "real", "tier_mode": "never"}
    assert parse_config("rewired-s0-fast-only") == {"base": "rewired-s0", "brain": "rewired-s0",
                                                    "tier_mode": "never"}


def test_scorecard_summary_call_stats_and_gated_pairing():
    from moai_echomind_fly.loop.scorecard import judge, paired, summarize

    def ep(seed, reached, calls, dec=10):
        return {"seed": seed, "reached": reached, "reach_ms": 1000.0 if reached else None,
                "final_food_mm": 1.0, "looming": False, "looming_hit": None, "decisions": dec,
                "violations": 0, "latency_s": 3.0, "llm_calls": calls}

    s = summarize([ep(1, True, 2), ep(2, False, 3)])
    assert s["llm_calls"] == 5 and s["call_rate"] == 0.25 and s["est_llm_wall_s"] == 15.0
    old = {k: v for k, v in s.items() if k not in ("llm_calls", "call_rate", "est_llm_wall_s")}
    s2 = summarize([{k: v for k, v in e.items() if k != "llm_calls"}
                    for e in (ep(1, True, 2), ep(2, False, 3))])
    assert s2 == old                                        # 옛 기록(칸 없음)은 옛 요약 그대로
    res = {"vector": [ep(i, True, 10) for i in range(4)],
           "vector-gated": [ep(i, True, 2) for i in range(4)],
           "rewired-s0-gated": [ep(i, False, 2) for i in range(4)],
           "fast-only": [ep(i, False, 0) for i in range(4)],
           "rewired-s0-fast-only": [ep(i, False, 0) for i in range(4)]}
    g = paired(res, "-gated")
    assert set(g) == {"vector", "rewired-s0"} and judge(g)["rewired"] == ["rewired-s0"]
    assert set(paired(res, "-fast-only")) == {"vector", "rewired-s0"}


def test_decision_row_carries_tier_fields_only_when_present():
    from moai_echomind_fly.loop.episode import Step
    from moai_echomind_fly.loop.scorecard import decision_row
    from moai_echomind_fly.loop.teacher import Truth

    t = Truth(False, 0.0, 0.0, False, odor_left=1.0, odor_right=1.0, odor_azimuth_deg=0.0,
              food_distance_mm=5.0, legs_in_contact=6)
    old = decision_row(1, Step(0, 0.0, t, "c", {}, {}, "vector", 0, (0.0, 0.0), {"raw": "x"}))
    assert "tier" not in old
    new = decision_row(1, Step(0, 0.0, t, "c", {}, {}, "fast", 0, (0.0, 0.0),
                               {"tier": "fast", "trigger": None, "fast_conf": 0.4}))
    assert new["tier"] == "fast" and new["fast_conf"] == 0.4


def test_forward_model_recovers_planted_dynamics_and_flags_a_jump():
    rng = np.random.default_rng(1)
    rows, xs = [], []
    for ep in range(40):
        z = rng.normal(size=4)
        for d in range(12):
            a = {"turn": ["left", "none", "right"][d % 3], "stop": False, "walk": 0.6}
            rows.append({"seed": ep, "decision": d, "action": a})
            xs.append(z.copy())
            z = 0.8 * z + 0.3 * fp.act_vec(a)[:4] + 0.01 * rng.normal(size=4)
    x = np.array(xs)
    fwd, _ = fp._fit_forward(x, rows, fp.FAST_RULE)
    pol = fp.FastPolicy({"forward": fwd}, ["a"], 0.5, {"rule": fp.FAST_RULE})
    ok = pol.surprise(x[0], rows[0]["action"], x[1])
    jump = pol.surprise(x[0], rows[0]["action"], x[1] + 3.0)
    assert ok < 0.01 < 1.0 < jump and pol.surprise(None, None, x[0]) == 0.0
    assert fp.split_episodes(list(range(40)), 0.3, 0) == fp.split_episodes(list(range(40)), 0.3, 0)
    assert len(fp.split_episodes(list(range(40)), 0.3, 0)) == 12


def test_matched_rate_and_gating_verdict(tmp_path: Path):
    import json

    from moai_echomind_fly.loop.scorecard import gating_verdict, matched_rate

    rows = [{"d": d, "tier": "llm" if d in (0, 2, 4) else "fast",
             "trigger": "goal" if d == 2 else ("ambiguous" if d == 4 else None)}
            for d in range(10)]
    f = tmp_path / "vector-gated.decisions.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert matched_rate(f, sniff=2) == pytest.approx(1 / 7)      # d 3~9 중 한 번

    def eps(vals):
        return [{"seed": i, "reached": bool(v)} for i, v in enumerate(vals)]

    n = 12
    res = {"vector": eps([1] * n), "fast-only": eps([0] * n), "vector-gated": eps([1] * n),
           "vector-random-matched": eps([0] * n)}
    v = gating_verdict(res)
    assert v["pgr"] == 1.0 and v["vector-random-matched"]["gated_beats"] is True
    res["vector-random-matched"] = eps([1] * n)
    assert "쓰지 않는다" in gating_verdict(res)["wording"]
    assert gating_verdict({"vector": eps([1])}) is None

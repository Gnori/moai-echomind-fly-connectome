"""두 층 제어 · 빠른 정책 (R-1027 · P loop Task 1 · 2) — 게이트 조건 · 정직 · 회귀(always)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from moai_echomind_fly.loop import fast_policy as fp
from moai_echomind_fly.loop.tiered import Gate, TieredPolicy

TH = {"max_gap_decisions": 5, "surprise_q": 0.9, "confidence_min": 0.3, "conflict_min": 0.6,
      "reflex_threat_min": None}
DQ = {"0.90": 1.0}


class StubFast:
    """z[0] = 돌기 확신도 · z[1] = 위험 점수 — 게이트가 읽는 값만 흉내."""

    def predict(self, z):
        a = {"walk": 0.6, "turn": "left", "turn_strength": 0.5, "stop": False, "groom": False}
        t = {"walk": 1.0, "turn": "right", "turn_strength": 1.0, "stop": False, "groom": False}
        return fp.FastOut(a, "left", float(z[0]), float(z[1]), False, t)


class StubLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, vectors, truth, obs, d):
        self.calls += 1
        return {"walk": 0.8, "turn": "none", "turn_strength": 0.0, "stop": False,
                "groom": False}, {"source": "vector"}


def _policy(mode, **kw):
    gate = Gate({"mode": mode, "thresholds": TH}, DQ if mode == "gated" else None,
                sniff_decisions=kw.pop("sniff", 0))
    llm = StubLLM() if mode != "never" else None
    return TieredPolicy(StubFast(), llm, gate, lambda v: np.asarray(v, float), **kw), llm


def test_always_calls_llm_every_decision_like_today():
    pol, llm = _policy("always")
    tiers = [pol([0.9, 0.0], None, None, d)[1]["tier"] for d in range(6)]
    assert tiers == ["llm"] * 6 and llm.calls == 6


def test_never_is_fast_only():
    pol, _ = _policy("never")
    out = [pol([0.9, 0.0], None, None, d) for d in range(4)]
    assert all(i["tier"] == "fast" for _, i in out) and out[0][0]["turn"] == "left"


def test_gated_triggers_goal_gap_ambiguous_conflict_surprise():
    pol, llm = _policy("gated")
    calm = [0.9, 0.0]
    got = [pol(calm, None, None, 0)[1]["trigger"]]            # 첫 결정 = 목표
    got += [pol(calm, None, None, d)[1]["trigger"] for d in (1, 2, 3, 4)]
    got.append(pol(calm, None, None, 5)[1]["trigger"])        # 마지막 호출 뒤 5 결정 = 시간 상한
    got.append(pol([0.1, 0.0], None, None, 6)[1]["trigger"])  # 확신도 낮음(특징 변화 0.8 < 1)
    got.append(pol([0.9, 0.9], None, None, 7)[1]["trigger"])  # 위험 · 먹이 쪽 둘 다 강함
    got.append(pol([5.0, 0.0], None, None, 8)[1]["trigger"])  # 특징이 크게 바뀜(놀람)
    assert got == ["goal", None, None, None, None, "gap", "ambiguous", "conflict", "surprise"]
    assert llm.calls == 5


def test_gated_refuses_unlocked_thresholds():
    with pytest.raises(ValueError, match="문턱 미정"):
        Gate({"mode": "gated", "thresholds": {**TH, "confidence_min": None}}, DQ)


def test_truth_is_never_read_by_the_tiers():
    """참값을 망가뜨려도 층 결정이 같다(정직 · P §1.2) — LLM 은 참값을 받기만."""
    a, _ = _policy("gated")
    b, _ = _policy("gated")
    feats = [[0.9, 0.0], [0.9, 0.0], [0.1, 0.0], [0.9, 0.9], [5.0, 0.0]]
    ta = [a(z, {"food": 1}, None, d)[1]["tier"] for d, z in enumerate(feats)]
    tb = [b(z, "쓰레기", object(), d)[1]["tier"] for d, z in enumerate(feats)]
    assert ta == tb


def test_hold_keeps_last_llm_action_and_reflex_skips_llm():
    pol, llm = _policy("gated", between_calls="hold")
    first = pol([0.9, 0.0], None, None, 0)[0]
    assert pol([0.9, 0.0], None, None, 1)[0] == first        # 부르지 않은 결정은 마지막 LLM 행동
    rx, _ = _policy("gated", reflex_threat_min=0.8)
    act, info = rx([0.9, 0.95], None, None, 0)
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

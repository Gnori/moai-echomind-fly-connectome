"""채점 평가 eval-v2 — 칸 만들기 · 갈라지는 토큰 · 예측 · 해당 표본만 세는 점수.

가짜 토크나이저(글자 = 토큰) · Qwen 없음.
"""

from __future__ import annotations

import numpy as np

from moai_echomind_fly.loop import caption_score as cs

CAP_OBJ = ("다가오는 물체: 왼쪽 · 가까움 · 다가옴 / 먹이 냄새: 오른쪽 · 강함 / 다리 접촉: 6개 / "
           "소리: 없음 / 암컷 대용 물체: 없음")
CAP_NONE = ("다가오는 물체: 없음 / 먹이 냄새: 없음 / 다리 접촉: 5개 / 소리: 있음 / "
            "암컷 대용 물체: 앞")
PREFIX_LEGS = "다가오는 물체: 왼쪽 · 가까움 · 다가옴 / 먹이 냄새: 오른쪽 · 강함 / 다리 접촉:"


def _encode(text: str) -> list[int]:
    """글자 하나 = 토큰 하나(공백 포함) — 접두가 늘 안정하다."""
    return [ord(c) for c in text]


def test_slots_follow_the_truth_and_skip_absent_subfields():
    names = [s.name for s in cs.slots_for(CAP_OBJ)]
    assert names == ["loom_head", "loom_distance", "loom_motion", "odor_head", "odor_level",
                     "legs", "sound", "female"]
    s = cs.slots_for(CAP_OBJ)
    assert s[1].prefix == "다가오는 물체: 왼쪽 ·" and s[1].truth == "가까움"
    assert s[3].prefix.endswith("/ 먹이 냄새:") and s[3].truth == "오른쪽"
    assert s[5].truth == "6개" and s[7].truth == "없음"
    none = [x.name for x in cs.slots_for(CAP_NONE)]
    assert none == ["loom_head", "odor_head", "legs", "sound", "female"]


def test_discriminating_token_handles_shared_leading_tokens():
    # " 0개" … " 6개" — 공백이 공통이라 두 번째 토큰(숫자)에서 갈라진다
    pos, ids = cs.discriminating_tokens(_encode, "다리 접촉:", cs.LEGS)
    assert pos == len("다리 접촉:") + 1 and ids == [ord(str(i)) for i in range(7)]
    pos, ids = cs.discriminating_tokens(_encode, "소리:", cs.YESNO)
    assert pos == len("소리:") + 1 and ids == [ord("있"), ord("없")]


def test_predict_picks_the_candidate_with_the_highest_logit_at_its_slot():
    truth_tokens = _encode(CAP_OBJ)
    vocab = 0xD7A4

    def logits_at(j: int) -> np.ndarray:
        row = np.full(vocab, -10.0)
        row[truth_tokens[j]] = 5.0            # 정답 토큰이 가장 높다 → 전부 맞힌다
        return row

    slots = cs.slots_for(CAP_OBJ)
    pred = cs.predict(slots, _encode, logits_at)
    assert pred == {s.name: s.truth for s in slots}

    def wrong_at(j: int) -> np.ndarray:        # 다리 자리에서 "3" 이 더 높다
        row = logits_at(j)
        if j == len(_encode(PREFIX_LEGS)) + 1:
            row[ord("3")] = 9.0
        return row

    pred2 = cs.predict(slots, _encode, wrong_at)
    assert pred2["legs"] == "3개" and pred2["loom_head"] == "왼쪽"


def test_score_counts_only_applicable_samples_and_majority_on_the_same_subset():
    s_obj, s_none = cs.slots_for(CAP_OBJ), cs.slots_for(CAP_NONE)
    t1 = cs.to_fields(s_obj, {s.name: s.truth for s in s_obj})
    t2 = cs.to_fields(s_none, {s.name: s.truth for s in s_none})
    assert "loom_side" in t1 and "loom_side" not in t2 and t2["loom_present"] == "없음"
    p1 = dict(t1, loom_distance="멀리")                     # 거리만 틀림
    p2 = dict(t2, legs="6개")                               # 다리만 틀림
    sc = cs.score([t1, t2], [p1, p2])
    assert sc["loom_present"]["n"] == 2 and sc["loom_present"]["accuracy"] == 1.0
    assert sc["loom_distance"]["n"] == 1 and sc["loom_distance"]["accuracy"] == 0.0
    assert sc["legs"]["n"] == 2 and sc["legs"]["accuracy"] == 0.5
    assert sc["legs"]["majority_baseline"] == 0.5           # 5개 · 6개 → 최빈 하나 = 1/2
    assert "odor_side" in sc and sc["odor_side"]["n"] == 1

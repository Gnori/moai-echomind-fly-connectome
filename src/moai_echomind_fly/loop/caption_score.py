"""캡션 **채점 평가** eval-v2 (R-7469 F-005 · 2026-09-24 요구자 결정 B).

왜: 자유 생성 평가(eval-caption · v1)는 서식을 잰다 — train-v4 투영기는 값(왼쪽 · 강함 · 6개)은
맞히면서 "다가오는 물체: … / 먹이 냄새: …" 틀을 안 배워 채점기가 0 점을 줬다
(runs/projector-real-train-v4/eval-caption-final.json 예: `<state>left</state> <taste>strong</taste>
<touch>6</touch>`). 학습 손실은 칸 값 토큰에 있으니 평가도 그것을 잰다.

어떻게(빈칸 채우기 · 교사 강제): 정답 캡션을 그대로 넣어 한 번 앞으로 돌리고, 칸마다 **그 칸
앞까지는 정답**인 자리에서 다음 토큰 분포를 본다. 후보 값(없음/앞/왼쪽/오른쪽 · 가까움/중간/멀리 ·
다가옴/머묾 · 강함/약함 · 0~6개 · 있음/없음)이 **처음 갈라지는 토큰**의 로짓을 비교해 가장 높은
후보를 고른 것이 예측. 접두가 토큰 경계에서 안정하므로(실측) 후보 하나 = 토큰 하나 비교로 충분하다.
서식과 무관 · 결정적 · 뇌마다 같은 절차.

칸은 **해당 표본에서만** 센다 — 물체 쪽 · 거리 · 움직임은 물체가 있는 표본, 냄새 쪽 · 세기는 냄새가
있는 표본. (v1 은 없는 표본의 "-" == "-" 를 맞힌 것으로 세어 물체 쪽 0.726 이 찍기와 같았다.)
찍기 = 같은 부분집합의 최빈값.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RULE = {"version": "eval-v2.1-scored", "method": "cloze · truth-prefix · first-token argmax",
        "applicable_only": True,
        # v2.1(2026-09-24 · v4 결과 뒤 · 까닭): 물체 쪽 · 냄새 쪽을 "없음"까지 넣고 고르면
        # "있음/없음" 판단과 섞인다(v4 진짜: 물체 있는 표본에서 "없음"이라 해 쪽 0.078).
        # → 쪽은 **있다고 알려 준 채** 세 방향(앞 · 왼쪽 · 오른쪽)만 놓고 고른다.
        # 있음/없음은 따로(loom_present · odor_present).
        "side_given_present": True}

SIDES = ["없음", "앞", "왼쪽", "오른쪽"]
DIST = ["가까움", "중간", "멀리"]
MOTION = ["다가옴", "머묾"]
LEVEL = ["강함", "약함"]
LEGS = [f"{i}개" for i in range(7)]
YESNO = ["있음", "없음"]

FIELD_ORDER = ["loom_present", "loom_side", "loom_distance", "loom_motion",
               "odor_present", "odor_side", "odor_level", "legs", "sound", "female"]


@dataclass
class Slot:
    """채울 칸 하나 — 접두(정답 · 이 칸 앞까지) · 후보 · 정답 값."""

    name: str
    prefix: str
    candidates: list[str]
    truth: str
    derived: dict[str, str] = field(default_factory=dict)   # 후보 → 파생 칸 값(있음/없음)


def parse_truth(caption: str) -> dict[str, Any]:
    """정답 캡션(teacher.caption 서식) → 구조."""
    fields: dict[str, str] = {}
    for part in caption.split(" / "):
        key, _, val = part.partition(": ")
        fields[key.strip()] = val.strip()
    loom = [x.strip() for x in fields.get("다가오는 물체", "없음").split("·")]
    odor = [x.strip() for x in fields.get("먹이 냄새", "없음").split("·")]
    return {"loom": loom, "odor": odor, "legs": fields.get("다리 접촉", ""),
            "sound": fields.get("소리", ""), "female": fields.get("암컷 대용 물체", "")}


def slots_for(caption: str) -> list[Slot]:
    """정답 캡션에서 채점할 칸들(해당하는 것만) — 접두는 정답 글 그대로."""
    t = parse_truth(caption)
    out: list[Slot] = []
    p = "다가오는 물체:"
    out.append(Slot("loom_head", p, SIDES, t["loom"][0]))
    if t["loom"][0] != "없음":
        p = f"{p} {t['loom'][0]} ·"
        out.append(Slot("loom_distance", p, DIST, t["loom"][1]))
        p = f"{p} {t['loom'][1]} ·"
        out.append(Slot("loom_motion", p, MOTION, t["loom"][2]))
        p = f"{p} {t['loom'][2]}"
    else:
        p = f"{p} 없음"
    p = f"{p} / 먹이 냄새:"
    out.append(Slot("odor_head", p, SIDES, t["odor"][0]))
    if t["odor"][0] != "없음":
        p = f"{p} {t['odor'][0]} ·"
        out.append(Slot("odor_level", p, LEVEL, t["odor"][1]))
        p = f"{p} {t['odor'][1]}"
    else:
        p = f"{p} 없음"
    p = f"{p} / 다리 접촉:"
    out.append(Slot("legs", p, LEGS, t["legs"]))
    p = f"{p} {t['legs']} / 소리:"
    out.append(Slot("sound", p, YESNO, t["sound"]))
    p = f"{p} {t['sound']} / 암컷 대용 물체:"
    out.append(Slot("female", p, SIDES, t["female"]))
    return out


def discriminating_tokens(encode, prefix: str, candidates: list[str]) -> tuple[int, list[int]]:
    """후보들이 **처음 갈라지는 토큰** 자리(접두 토큰 수 + 공통 토큰 수)와 후보마다 그 자리의 토큰.

    접두 토큰이 그대로여야(경계 안정) 하고, 갈라지는 자리의 토큰은 서로 달라야 한다.
    실측(Qwen3.8 토크나이저): 쪽 · 거리 · 움직임 · 세기 · 있음/없음은 첫 토큰에서 갈라지고,
    다리 "N개" 는 " " 뒤 숫자 토큰에서 갈라진다(공통 1).
    """
    base = list(encode(prefix))
    tails = []
    for c in candidates:
        toks = list(encode(f"{prefix} {c}"))
        if toks[:len(base)] != base:
            raise ValueError(f"접두가 토큰 경계에서 흔들린다: {prefix!r} + {c!r}")
        tails.append(toks[len(base):])
    d = 0
    while all(len(t) > d for t in tails) and len({t[d] for t in tails}) == 1:
        d += 1
    if not all(len(t) > d for t in tails):
        raise ValueError(f"후보 하나가 다른 후보의 접두다: {candidates}")
    ids = [int(t[d]) for t in tails]
    if len(set(ids)) != len(ids):
        raise ValueError(f"갈라지는 자리의 토큰이 겹친다: {candidates} → {ids}")
    return len(base) + d, ids


def predict(slots: list[Slot], encode, logits_at) -> dict[str, str]:
    """칸마다 예측 — `logits_at(j)` = 정답 글 j 번째 토큰을 맞히는 자리의 로짓(1차원).

    쪽 칸(loom_head · odor_head)은 둘을 낸다: 후보 전부(없음 포함)에서 고른 것과,
    **없음을 뺀 세 방향**에서 고른 것(`<이름>_side` · v2.1 · 있다고 알려 준 채 쪽만 묻기).
    """
    pred: dict[str, str] = {}
    for s in slots:
        pos, ids = discriminating_tokens(encode, s.prefix, s.candidates)
        row = np.asarray(logits_at(pos), dtype=np.float64)
        pred[s.name] = s.candidates[int(np.argmax(row[ids]))]
        if s.candidates is SIDES:
            keep = [i for i, c in enumerate(s.candidates) if c != "없음"]
            pred[f"{s.name}_side"] = s.candidates[keep[int(np.argmax(row[[ids[i] for i in keep]]))]]
    return pred


def to_fields(slots: list[Slot], values: dict[str, str]) -> dict[str, str]:
    """(진짜 값 또는 예측) → 보고 칸. 없는 칸은 빠진다(해당 표본만 센다)."""
    out: dict[str, str] = {}
    head = values.get("loom_head")
    if head is not None:
        out["loom_present"] = "없음" if head == "없음" else "있음"
        if any(s.name == "loom_distance" for s in slots):        # 정답에 물체가 있는 표본
            out["loom_side"] = values.get("loom_head_side", head)   # v2.1: 있다고 알려 준 채 쪽만
            out["loom_distance"] = values["loom_distance"]
            out["loom_motion"] = values["loom_motion"]
    oh = values.get("odor_head")
    if oh is not None:
        out["odor_present"] = "없음" if oh == "없음" else "있음"
        if any(s.name == "odor_level" for s in slots):
            out["odor_side"] = values.get("odor_head_side", oh)
            out["odor_level"] = values["odor_level"]
    for k in ("legs", "sound", "female"):
        if k in values:
            out[k] = values[k]
    return out


def score(truths: list[dict[str, str]], preds: list[dict[str, str]]) -> dict[str, Any]:
    """칸마다 정확도 · 찍기(최빈값) · 표본 수 — 그 칸이 있는 표본만."""
    out: dict[str, Any] = {}
    for k in FIELD_ORDER:
        pairs = [(t[k], p[k]) for t, p in zip(truths, preds) if k in t and k in p]
        if not pairs:
            continue
        major = Counter(t for t, _ in pairs).most_common(1)[0][0]
        out[k] = {"n": len(pairs),
                  "accuracy": float(np.mean([t == p for t, p in pairs])),
                  "majority_baseline": float(np.mean([t == major for t, _ in pairs])),
                  "majority": major}
    return out

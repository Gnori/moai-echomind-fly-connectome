"""글과 초파리 신호를 섞은 입력 (R-3340 F-013 · F-017 · I §3.2).

요구자: "일반 텍스트도 들어가서 · 초파리 신호도 들어가게" —
이미지 읽는 LLM 이 글과 이미지를 섞어 받듯, 글 사이에 채널 태그와 자리 표시 토큰
`<|vision_pad|>`(Qwen 이 원래 이미지 임베딩을 끼우는 자리)를 둔다.
"""

from __future__ import annotations

import json

SLOT = "<|vision_pad|>"

SYSTEM_ACTION = (
    "너는 초파리의 뇌다. "
    "아래 <채널> 사이에는 초파리 회로가 지난 300ms 동안 낸 신경 신호가 들어 있다. "
    "목표: 먹이 냄새 쪽으로 가되, 다가오는 물체는 피한다. "
    'JSON 한 줄로만 답한다: {"walk":0~1,"turn":"left|right|none","turn_strength":0~1,'
    '"stop":true|false,"groom":true|false}'
)
SYSTEM_CAPTION = (
    "너는 초파리의 뇌다. "
    "아래 <채널> 사이에는 초파리 회로가 지난 300ms 동안 낸 신경 신호가 들어 있다. "
    "지금 무엇이 느껴지는지 정해진 서식으로 한 줄로 답한다."
)
INSTRUCTION_ACTION = "지금 어떻게 움직일까?"
INSTRUCTION_CAPTION = "지금 무엇이 느껴지나?"


def signal_block(channels: list[str], k: int) -> str:
    return " ".join(f"<{c}>{SLOT * k}</{c}>" for c in channels)


def messages(system: str, instruction: str, channels: list[str], k: int,
             text_summary: str | None = None) -> list[dict[str, str]]:
    """벡터 방식이면 자리 표시 토큰, 글 방식(비교군)이면 요약 글이 같은 자리에 들어간다."""
    body = text_summary if text_summary is not None else signal_block(channels, k)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{instruction}\n{body}"}]


def action_text(action: dict) -> str:
    keys = ["walk", "turn", "turn_strength", "stop", "groom"]
    return json.dumps({k: action[k] for k in keys}, ensure_ascii=False, separators=(",", ":"))

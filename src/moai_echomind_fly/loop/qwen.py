"""Qwen3.8-27B 를 우리 프로세스에서 직접 (R-3340 F-013 · I §3.2).

서버(oMLX · Ollama)를 쓰지 않는다. mlx-lm 의 공개 API 만 쓴다:
`load` · `generate_step(..., input_embeddings=)` · 모델의 `input_embeddings` 인자.
토큰 대신 같은 토큰의 임베딩을 넣으면 로짓이 정확히 같다(2026-09-22 실측 차이 0.0) —
그래서 글 토큰 사이
자리 표시 칸에 초파리 신호 벡터(투영기 출력)를 끼워 넣을 수 있다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    gen_tokens: int
    prefill_s: float
    decode_s: float

    @property
    def decode_tok_s(self) -> float:
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0


class QwenLM:
    """가중치를 한 번 올리고 여러 번 쓴다. 가중치는 얼린 채다(학습은 투영기만 — F-017)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.cfg = cfg or config.load("llm")
        self.mx = mx
        t0 = time.time()
        self.model, self.tok = load(str(Path(self.cfg["weights"]).expanduser()))
        mx.eval(self.model.parameters())
        self.model.freeze()
        self.load_s = time.time() - t0
        self.lm = self.model.language_model
        self.embed = self.lm.model.embed_tokens
        self.hidden = int(self.lm.args.hidden_size)
        self.label = self.cfg["label"]

    # ------------------------------------------------------------------ 글
    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
               ) -> list[int]:
        return list(self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            enable_thinking=bool(self.cfg["enable_thinking"])))

    def generate(self, prompt: list[int], *, max_tokens: int | None = None,
                 input_embeddings=None) -> Generation:
        """탐욕 복호(온도 0) — 같은 입력이면 같은 출력."""
        from mlx_lm.generate import generate_step

        mx = self.mx
        limit = int(max_tokens or self.cfg["max_tokens"])
        ids = mx.array(prompt)
        out: list[int] = []
        t0 = time.time()
        t_first = None
        steps = generate_step(ids, self.model, max_tokens=limit, input_embeddings=input_embeddings)
        for (token, _), _ in zip(steps, range(limit)):
            if t_first is None:
                t_first = time.time()
            t = int(token)
            if t in self.tok.eos_token_ids:
                break
            out.append(t)
        t_end = time.time()
        t_first = t_first or t_end
        return Generation(self.tok.decode(out), len(prompt), len(out),
                          t_first - t0, t_end - t_first)

    # ------------------------------------------------------------------ 벡터 주입
    def token_embeddings(self, prompt: list[int]):
        return self.embed(self.mx.array(prompt))

    def splice(self, prompt: list[int], slot_token: int, soft):
        """`slot_token` 자리들을 순서대로 `soft`[n_slots, hidden] 로 바꾼 임베딩 [L, hidden]."""
        mx = self.mx
        ids = mx.array(prompt)
        emb = self.embed(ids)
        pos = [i for i, t in enumerate(prompt) if t == slot_token]
        if len(pos) != soft.shape[0]:
            raise ValueError(f"자리 {len(pos)} 개 ≠ 소프트 토큰 {soft.shape[0]} 개")
        mask = mx.array([t == slot_token for t in prompt])[:, None]
        order = mx.cumsum(mask[:, 0].astype(mx.int32)) - 1
        filled = soft[mx.maximum(order, 0)].astype(emb.dtype)
        return mx.where(mask, filled, emb)


# ---------------------------------------------------------------------- 도구 호출 읽기
_TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str) -> Any:
    v = value.strip()
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Qwen3.8 채팅 틀의 형식 `<tool_call><function=…><parameter=…>…` 을 읽는다."""
    calls = []
    for name, body in _TOOL_RE.findall(text):
        args = {k: _coerce(v) for k, v in _PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls


def text_before_tool_call(text: str) -> str:
    return text.split("<tool_call>", 1)[0].strip()

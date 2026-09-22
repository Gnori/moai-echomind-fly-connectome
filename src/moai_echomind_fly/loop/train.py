"""투영기 학습 (R-3340 F-017) — Qwen 은 얼리고 투영기만.

손실 = 정답 글(캡션 또는 행동 JSON) 토큰의 교차 엔트로피.
기울기는 Qwen 을 거꾸로 지나 투영기로만 간다.
Qwen3.8 의 선형 어텐션 층(GatedDeltaNet)은 `training` 이면 전용 커널 대신
미분 가능한 연산 경로를 탄다
(mlx-lm `qwen3_5.py`: `use_kernel=not self.training`) — 그래서 학습 중에는 모델을 학습 모드로 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .fly_prompt import SLOT
from .projector import ChannelProjector
from .qwen import QwenLM


@dataclass
class Sample:
    prompt: list[int]                  # 자리 표시 토큰이 든 입력(채팅 틀 적용 · 생성 시작 표시까지)
    target: list[int]                  # 정답 글 토큰 + 끝 표시
    feats: dict[str, Any]              # 채널 → 특징 벡터(np.float32)


class Trainer:
    def __init__(self, qwen: QwenLM, projector: ChannelProjector, lr: float = 1e-4,
                 checkpoint: bool = True, cache_limit_gb: float | None = None) -> None:
        self.qwen = qwen
        if cache_limit_gb is not None:
            # 캐시 상한 — 두지 않으면 걸음마다 쌓여 13 시간 뒤 103 GB 가 됐다(2026-09-22)
            mx.set_cache_limit(int(cache_limit_gb * 1e9))
        if checkpoint:
            # 층마다 역전파 때 다시 계산해 저장을 줄인다
            # — mlx-lm 이 자체 LoRA 학습에서 쓰는 공개 함수.
            # 끄면 최대 메모리 92.9 GB(2026-09-22 실측 · 128 GB 맥에서 위험).
            from mlx_lm.tuner.trainer import grad_checkpoint

            grad_checkpoint(qwen.lm.model.layers[0])
        self.proj = projector
        self.slot = qwen.tok.convert_tokens_to_ids(SLOT)
        self.opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)
        self._step = nn.value_and_grad(self.proj, self._loss)

    def _hidden(self, emb: mx.array) -> mx.array:
        return self.qwen.lm.model(None, input_embeddings=emb[None])[0]

    def _loss(self, s: Sample) -> mx.array:
        soft = self.proj({k: mx.array(v) for k, v in s.feats.items()})
        prompt_emb = self.qwen.splice(s.prompt, self.slot, soft)
        tgt_emb = self.qwen.embed(mx.array(s.target[:-1]))
        h = self._hidden(mx.concatenate([prompt_emb, tgt_emb], axis=0))
        h = h[len(s.prompt) - 1:]                      # 정답 토큰을 맞히는 자리만
        logits = self.qwen.lm.lm_head(h)
        return nn.losses.cross_entropy(logits, mx.array(s.target), reduction="mean")

    def step(self, s: Sample) -> float:
        self.qwen.model.train()                        # 미분 가능한 경로(모듈 설명)
        loss, grads = self._step(s)
        self.opt.update(self.proj, grads)
        mx.eval(self.proj.parameters(), self.opt.state, loss)
        mx.clear_cache()                               # 걸음마다 비운다(위 캐시 상한 설명)
        return float(loss)

    def loss_only(self, s: Sample) -> float:
        self.qwen.model.eval()
        loss = self._loss(s)
        mx.eval(loss)
        return float(loss)

    def generate(self, s: Sample, max_tokens: int = 80) -> str:
        self.qwen.model.eval()
        soft = self.proj({k: mx.array(v) for k, v in s.feats.items()})
        emb = self.qwen.splice(s.prompt, self.slot, soft)
        return self.qwen.generate(s.prompt, max_tokens=max_tokens, input_embeddings=emb).text

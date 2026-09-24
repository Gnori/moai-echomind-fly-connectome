"""투영기 학습 (R-3340 F-017) — Qwen 은 얼리고 투영기만.

손실 = 정답 글(캡션 또는 행동 JSON) 토큰의 교차 엔트로피.
기울기는 Qwen 을 거꾸로 지나 투영기로만 간다.
Qwen3.8 의 선형 어텐션 층(GatedDeltaNet)은 `training` 이면 전용 커널 대신
미분 가능한 연산 경로를 탄다
(mlx-lm `qwen3_5.py`: `use_kernel=not self.training`) — 그래서 역전파 때는 모델을 학습 모드로 둔다.

그 경로는 토큰마다 상태(층당 3 MB)를 남겨 한 번에 미분하면 **토큰당 약 0.32 GB** 가 든다
(2026-09-22 실측: 226 토큰 → 최대 102 GB · 191 토큰 → 90 GB).
mlx-lm 의 층별 재계산(`grad_checkpoint`)은 전체를 한 번에 평가하는 지연 평가에서 듣지 않았다
(켜도 끄도 90~100 GB).
그래서 **한 층씩 거꾸로** 미분하고 층마다 평가해 끊는다 — 한 번에 한 층 몫만 든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .fly_prompt import SLOT
from .projector import ChannelProjector
from .qwen import QwenLM


@dataclass
class Sample:
    prompt: list[int]                  # 자리 표시 토큰이 든 입력(채팅 틀 적용 · 생성 시작 표시까지)
    target: list[int]                  # 정답 글 토큰 + 끝 표시
    feats: dict[str, Any]              # 채널 → 특징 벡터(np.float32)
    weights: list[float] | None = None  # 정답 토큰마다 손실 가중(train-v4) · 없으면 고르게


class Trainer:
    def __init__(self, qwen: QwenLM, projector: ChannelProjector, lr: float = 1e-4,
                 cache_limit_gb: float | None = None) -> None:
        self.qwen = qwen
        if cache_limit_gb is not None:
            # 캐시 상한 — 두지 않으면 걸음마다 쌓여 13 시간 뒤 103 GB 가 됐다(2026-09-22)
            mx.set_cache_limit(int(cache_limit_gb * 1e9))
        self.proj = projector
        self.slot = qwen.tok.convert_tokens_to_ids(SLOT)
        self.lr = lr
        self.reset_optimizer()
        self._pull = nn.value_and_grad(self.proj, self._pulled)

    def reset_optimizer(self) -> None:
        self.opt = optim.AdamW(learning_rate=self.lr, weight_decay=0.01)

    def _embeddings(self, s: Sample) -> mx.array:
        soft = self.proj({k: mx.array(v) for k, v in s.feats.items()})
        prompt_emb = self.qwen.splice(s.prompt, self.slot, soft)
        tgt_emb = self.qwen.embed(mx.array(s.target[:-1]))
        return mx.concatenate([prompt_emb, tgt_emb], axis=0)

    def _head_loss(self, h: mx.array, s: Sample) -> mx.array:
        h = self.qwen.lm.model.norm(h)[0][len(s.prompt) - 1:]   # 정답 토큰을 맞히는 자리만
        return self._ce(self.qwen.lm.lm_head(h), s)

    @staticmethod
    def _ce(logits: mx.array, s: Sample) -> mx.array:
        if s.weights is None:
            return nn.losses.cross_entropy(logits, mx.array(s.target), reduction="mean")
        w = mx.array(s.weights, dtype=mx.float32)
        ce = nn.losses.cross_entropy(logits, mx.array(s.target), reduction="none")
        return (ce * w).sum() / w.sum()

    def _loss(self, s: Sample) -> mx.array:
        h = self.qwen.lm.model(None, input_embeddings=self._embeddings(s)[None])
        h = h[0][len(s.prompt) - 1:]
        return self._ce(self.qwen.lm.lm_head(h), s)

    def _pulled(self, s: Sample, cot: mx.array) -> mx.array:
        # ⟨임베딩(투영기), 임베딩 쪽 기울기⟩ 의 기울기 = 투영기 기울기
        return (self._embeddings(s).astype(mx.float32) * cot).sum()

    def grads(self, s: Sample) -> tuple[mx.array, Any]:
        """손실과 투영기 기울기 — 층마다 평가해 끊는 역전파(모듈 설명)."""
        m = self.qwen.lm.model
        x = mx.stop_gradient(self._embeddings(s))[None]
        fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
        masks = [ssm if layer.is_linear else fa for layer in m.layers]
        self.qwen.model.eval()                         # 앞으로는 전용 커널(빠르다)
        inputs = []
        for layer, mask in zip(m.layers, masks):
            inputs.append(x)
            x = layer(x, mask=mask)
            mx.eval(x)
        loss, cot = mx.value_and_grad(lambda h: self._head_loss(h, s))(x)
        mx.eval(loss, cot)
        self.qwen.model.train()                        # 거꾸로는 미분 가능한 경로
        for layer, mask, xi in zip(reversed(m.layers), reversed(masks), reversed(inputs)):
            _, (cot,) = mx.vjp(lambda h: layer(h, mask=mask), [xi], [cot])
            mx.eval(cot)
        _, grads = self._pull(s, cot[0].astype(mx.float32))
        return loss, grads

    def step(self, s: Sample) -> float:
        loss, grads = self.grads(s)
        self.opt.update(self.proj, grads)
        mx.eval(self.proj.parameters(), self.opt.state, loss)
        mx.clear_cache()                               # 걸음마다 비운다(위 캐시 상한 설명)
        return float(loss)

    def loss_only(self, s: Sample) -> float:
        self.qwen.model.eval()
        loss = self._loss(s)
        mx.eval(loss)
        return float(loss)

    def target_logits(self, s: Sample) -> mx.array:
        """정답 자리마다 로짓 [len(target), vocab] — j 행이 target[j] 를 맞히는 자리(eval-v2)."""
        self.qwen.model.eval()
        h = self.qwen.lm.model(None, input_embeddings=self._embeddings(s)[None])
        logits = self.qwen.lm.lm_head(h[0][len(s.prompt) - 1:]).astype(mx.float32)
        mx.eval(logits)                                # float32 — numpy 가 bfloat16 을 못 읽는다
        return logits

    def generate(self, s: Sample, max_tokens: int = 80) -> str:
        self.qwen.model.eval()
        soft = self.proj({k: mx.array(v) for k, v in s.feats.items()})
        emb = self.qwen.splice(s.prompt, self.slot, soft)
        return self.qwen.generate(s.prompt, max_tokens=max_tokens, input_embeddings=emb).text

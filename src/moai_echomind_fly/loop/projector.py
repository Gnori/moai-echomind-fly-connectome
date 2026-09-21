"""투영기 (R-3340 F-017 · I §3.2) — 초파리 채널 벡터 → Qwen 은닉 5,120 차원 소프트 토큰.

구도: 초파리 회로 = 인코더(앞) · **투영기 = 다리** · Qwen = 생성기(뒤). 이미지 읽는 LLM 의 투영기
(LLaVA)와 같은 자리이고, 이미지 대신 초파리 뉴런 활동이 들어간다(요구자 그림).
학습되는 것은 이것뿐이다.

채널마다 하나: 벡터 → (선형 | 2층 MLP) → k 개 토큰. 출력은 토큰마다 RMS 를 Qwen 토큰 임베딩의
평균 RMS 에 맞춘다 — 소프트 토큰이 글 토큰과 같은 크기로 들어가게.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class ChannelProjector(nn.Module):
    def __init__(self, dims: dict[str, int], *, hidden: int, k: int, kind: str = "mlp2",
                 width: int = 256, target_rms: float = 1.0) -> None:
        super().__init__()
        self.names = list(dims)
        self.k = k
        self.hidden = hidden
        self.kind = kind
        self.target_rms = target_rms
        if kind == "linear":
            self.layers = [nn.Linear(d, k * hidden) for d in dims.values()]
        elif kind == "mlp2":
            self.layers = [
                nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, k * hidden))
                for d in dims.values()]
        else:
            raise ValueError(f"kind = linear | mlp2 (받은 값 {kind})")

    def __call__(self, feats: dict[str, mx.array]) -> mx.array:
        """채널 순서대로 이어 붙인 소프트 토큰 [채널 수 × k, hidden]."""
        out = []
        for name, layer in zip(self.names, self.layers):
            y = layer(feats[name]).reshape(self.k, self.hidden)
            rms = mx.sqrt(mx.mean(y * y, axis=-1, keepdims=True) + 1e-6)
            out.append(y / rms * self.target_rms)
        return mx.concatenate(out, axis=0)

    def n_params(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))

    def meta(self) -> dict[str, Any]:
        return {"names": self.names, "k": self.k, "hidden": self.hidden, "kind": self.kind,
                "target_rms": self.target_rms, "params": self.n_params()}

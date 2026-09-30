"""층별 역전파(`Trainer.grads`)가 한 번에 미분한 기울기와 같은가 — 작은 가짜 Qwen3.5 로.

실제 모델(27B)을 올리지 않는다. 선형 어텐션 층 3 · 전체 어텐션 층 1 로 같은 구조만 흉내 낸다.
CPU 에서 잰다 — GPU 는 평가를 나누는 방식에 따라 계산 순서가 달라 기울기가 0.2 % 까지 흔들린다
(같은 방식끼리는 0 · CPU 에서 두 방식 차이 1e-6 · 2026-09-22 실측).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from moai_echomind_fly.loop.fly_prompt import SLOT
from moai_echomind_fly.loop.projector import ChannelProjector
from moai_echomind_fly.loop.qwen import QwenLM
from moai_echomind_fly.loop.train import Sample, Trainer

SLOT_ID = 7


class _Tok:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == SLOT
        return SLOT_ID


def _tiny_qwen() -> QwenLM:
    mx.random.seed(0)
    lm = TextModel(TextModelArgs(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=300,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=64,
        linear_value_head_dim=32, full_attention_interval=4))
    mx.eval(lm.parameters())
    lm.freeze()
    q = QwenLM.__new__(QwenLM)                     # 가중치 파일 없이 필요한 칸만
    q.mx, q.model, q.lm, q.tok = mx, lm, lm, _Tok()
    q.embed = lm.model.embed_tokens
    q.hidden = 64
    return q


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
def test_layerwise_grads_match_full_graph(chunk):
    """층을 k 개씩 묶어도(D 14) 한 번에 미분한 것과 같다 — 3 은 나누어떨어지지 않는 경우."""
    mx.set_default_device(mx.cpu)
    try:
        _check_layerwise(chunk)
    finally:
        mx.set_default_device(mx.gpu)


def _check_layerwise(chunk: int = 1):
    q = _tiny_qwen()
    rng = np.random.default_rng(0)
    dims = {"smell": 12, "vision": 20}
    proj = ChannelProjector(dims, hidden=64, k=2, kind="mlp2", width=16, target_rms=0.02)
    mx.eval(proj.parameters())
    trainer = Trainer(q, proj, backprop_chunk=chunk)
    prompt = [11, 12, 13] + [SLOT_ID] * 4 + [14, 15, 16, 17]
    s = Sample(prompt, [21, 22, 23, 24],
               {n: rng.standard_normal(d).astype(np.float32) for n, d in dims.items()})

    loss, grads = trainer.grads(s)
    q.model.train()
    ref_loss, ref = nn.value_and_grad(proj, trainer._loss)(s)

    assert abs(float(loss) - float(ref_loss)) < 1e-5
    got, want = dict(tree_flatten(grads)), dict(tree_flatten(ref))
    assert got.keys() == want.keys()
    for k in want:
        assert float(mx.abs(want[k]).max()) > 0, k           # 기울기가 실제로 흐른다
        scale = float(mx.abs(want[k]).max())               # 텐서 크기 대비 오차
        np.testing.assert_allclose(np.array(got[k]), np.array(want[k]), rtol=0, atol=1e-5 * scale)


def test_weighted_loss_equals_mean_when_uniform_and_ignores_zero_weights():
    logits = mx.array(np.random.default_rng(0).standard_normal((4, 10)).astype(np.float32))
    tgt = [1, 2, 3, 4]
    plain = Trainer._ce(logits, Sample([0], tgt, {}))
    same = Trainer._ce(logits, Sample([0], tgt, {}, [1.0, 1.0, 1.0, 1.0]))
    assert abs(float(plain) - float(same)) < 1e-6
    only_first = Trainer._ce(logits, Sample([0], tgt, {}, [1.0, 0.0, 0.0, 0.0]))
    ce0 = float(nn.losses.cross_entropy(logits[:1], mx.array(tgt[:1]), reduction="mean"))
    assert abs(float(only_first) - ce0) < 1e-6

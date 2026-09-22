"""C 단계 부품 — 무거운 자원 없이 도는 시험(교사 · 디코더 · 채널 표 · 과제 뽑기)."""

from __future__ import annotations

import numpy as np
import pytest

from moai_echomind_fly import config, graph
from moai_echomind_fly.loop import teacher
from moai_echomind_fly.loop.decoder import Decoder, parse_action
from moai_echomind_fly.loop.world import make_task


def _truth(**kw) -> teacher.Truth:
    base = dict(looming_present=False, looming_azimuth_deg=0.0, looming_size_deg=0.0,
                looming_approaching=False, odor_left=1.0, odor_right=1.0, odor_azimuth_deg=0.0,
                food_distance_mm=20.0, legs_in_contact=6)
    base.update(kw)
    return teacher.Truth(**base)


def test_teacher_avoids_threat_before_food():
    t = _truth(looming_present=True, looming_azimuth_deg=40.0, looming_size_deg=25.0,
               looming_approaching=True, odor_azimuth_deg=40.0)
    assert teacher.action(t)["turn"] == "right"          # 왼쪽 위협 → 오른쪽으로
    assert "왼쪽 · 중간 · 다가옴" in teacher.caption(t)


def test_teacher_goes_to_food_and_stops():
    assert teacher.action(_truth(odor_azimuth_deg=-50.0))["turn"] == "right"
    assert teacher.action(_truth(odor_azimuth_deg=5.0))["turn"] == "none"
    assert teacher.action(_truth(food_distance_mm=1.0))["stop"] is True


@pytest.mark.parametrize("text, ok", [
    ('{"walk":0.7,"turn":"left","turn_strength":0.5,"stop":false,"groom":false}', True),
    ('설명 먼저 {"walk":2,"turn":"none","stop":false}', True),          # 범위는 자른다
    ('{"walk":0.7,"turn":"up","stop":false}', False),                    # turn 값
    ('{"walk":0.7,"turn":"left","stop":false,"jump":true}', False),      # 카탈로그 밖 키
    ('걷는다', False),
])
def test_parse_action_schema(text, ok):
    a, why = parse_action(text)
    assert (a is not None) == ok, why
    if a:
        assert 0.0 <= a["walk"] <= 1.0


def test_task_is_seeded():
    cfg = config.load("task")
    assert make_task(5, cfg) == make_task(5, cfg)


@pytest.mark.data
def test_decoder_catalog_and_channel_exclusion():
    from moai_echomind_fly.loop.encoder import build_map
    from moai_echomind_fly.loop.signals import ChannelTable
    from moai_echomind_fly.protocols import LoomSpec

    g = graph.load()
    cfg = config.load("task")
    dec = Decoder(g, cfg)
    assert set(g.columns["type"][dec.dn_idx]) == {"DNp09", "DNa02"}
    r = dec.target_rates({"walk": 0.5, "turn": "left", "turn_strength": 1.0, "stop": False,
                          "groom": False})
    assert r[dec.walk_pos].tolist() == [50.0, 50.0] and r[dec.turn_pos["L"]].tolist() == [100.0]
    smap = build_map(g, cfg, LoomSpec.from_config())
    assert all(len(v) > 100 for v in smap.touch.values())     # rootSide 로 좌우를 가른다
    table = ChannelTable.build(g, exclude=np.concatenate([smap.stim_idx, dec.dn_idx]))
    in_channels = np.concatenate([c.neurons for c in table.channels.values()])
    assert not np.isin(in_channels, smap.stim_idx).any()      # 직접 자극한 뉴런은 벡터에 없다
    assert not np.isin(in_channels, dec.dn_idx).any()


def test_linear_probe_finds_a_planted_signal():
    """심어 둔 신호는 찾고, 없는 것은 찍기 수준 — 검사 자체가 도는지 본다."""
    from moai_echomind_fly.loop.linear_probe import probe_field

    rng = np.random.default_rng(0)
    labels = ["왼쪽", "오른쪽"] * 40
    x = rng.standard_normal((80, 50))
    x[::2, 0] += 6.0                                   # 첫 칸에만 신호를 심는다
    tr, te = slice(0, 60), slice(60, 80)
    got = probe_field(x[tr], labels[:60], x[te], labels[60:])
    assert got["accuracy"] > 0.9 >= got["baseline"]
    noise = probe_field(rng.standard_normal((60, 50)), labels[:60],
                        rng.standard_normal((20, 50)), labels[60:])
    assert noise["accuracy"] <= 0.75                   # 신호가 없으면 못 맞힌다

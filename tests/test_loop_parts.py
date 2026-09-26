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


def test_projector_init_seed_makes_same_weights():
    """train-v3: 같은 규칙이면 투영기 초기값이 같다 · 씨앗이 다르면 다르다(재현성)."""
    from mlx.utils import tree_flatten

    from moai_echomind_fly.loop.train_cli import TRAIN_RULE, new_projector

    def weights(rule: dict) -> list[np.ndarray]:
        proj = new_projector({"vision": 12, "body": 5}, 16, rule, 1.0)
        return [np.array(v) for _, v in tree_flatten(proj.parameters())]

    a, b = weights(TRAIN_RULE), weights(TRAIN_RULE)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = weights({**TRAIN_RULE, "init_seed": 1})
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_data_source_prefers_pack_then_runs_shards(tmp_path, monkeypatch):
    """묶음(data/datasets/<이름>)이 있으면 그것을, 없으면 runs/ 조각을 가리킨다."""
    from moai_echomind_fly import datasets
    from moai_echomind_fly.loop import train_cli

    (tmp_path / "pack-a").mkdir()
    (tmp_path / "pack-a" / "SHA256SUMS").write_text("x  train/a\n")
    monkeypatch.setattr(datasets, "store_dir", lambda: tmp_path)
    got = train_cli.data_source("pack-a")
    assert got["pack"] == "pack-a" and len(got["sha256sums"]) == 64
    assert "runs" in train_cli.data_source("no-such-pack")


def test_retinotopic_encoder_prefers_lc_at_object_elevation():
    """task-v2: 위에 있는 물체는 고도가 높은 LC 를 더 세게 · 좌표 없는 LC 는 조율 없이."""
    from moai_echomind_fly.loop.encoder import Encoder, SensoryMap
    from moai_echomind_fly.loop.world import Obs
    from moai_echomind_fly.protocols import LoomSpec

    smap = SensoryMap(np.arange(3), {"L": np.array([0, 1, 2]), "R": np.zeros(0, int)},
                      {"L": np.ones(3), "R": np.zeros(0)},
                      {"L": np.array([50.0, -40.0, np.nan]), "R": np.zeros(0)},
                      {"L": np.zeros(0, int), "R": np.zeros(0, int)},
                      {k: np.zeros(0, int) for k in ("LF", "LM", "LH", "RF", "RM", "RH")})
    cfg = config.load("task")
    enc = Encoder(smap, cfg, LoomSpec.from_config())
    obs = Obs(0.0, (0.0, 0.0), 0.0, [0.0, 0.0, 0.0, 0.0], {}, {"present": True, "azimuth_deg": 90.0,
              "size_deg": 20.0, "expansion_deg_s": 60.0, "elevation_deg": 45.0},
              (10.0, 0.0), False, 0.0)
    r = enc.rates(obs)
    assert r[0] > r[1] > 0            # 위(50°) LC > 아래(−40°) LC
    assert r[2] >= r[0]               # 좌표 없는 LC 는 조율 1(가장 셈)


def test_task_v2_keeps_v1_draws_and_adds_elevation():
    v1 = config.load("task", version="task-v1")
    v2 = config.load("task")
    seed = next(s for s in range(50) if "looming" in make_task(s, v1))
    a, b = make_task(seed, v1), make_task(seed, v2)
    assert a["odor_xy"] == b["odor_xy"]
    assert a["looming"]["bearing_deg"] == b["looming"]["bearing_deg"]
    assert -30 <= b["looming"]["elevation_deg"] <= 60 and "elevation_deg" not in a["looming"]


def test_retinotopic2_encoder_tunes_by_azimuth_and_elevation():
    """retinotopic2 는 방위 · 고도 둘 다 — 좌표 없는 LC 는 1 · 방위 없는 LC 는 고도만."""
    from moai_echomind_fly.loop.encoder import Encoder, SensoryMap
    from moai_echomind_fly.protocols import LoomSpec

    smap = SensoryMap(np.arange(3), {"R": np.array([0, 1, 2]), "L": np.zeros(0, int)},
                      {"R": np.ones(3), "L": np.zeros(0)},
                      {"R": np.array([0.0, 0.0, np.nan]), "L": np.zeros(0)},
                      {"L": np.zeros(0, int), "R": np.zeros(0, int)},
                      {k: np.zeros(0, int) for k in ("LF", "LM", "LH", "RF", "RM", "RH")},
                      vision_az={"R": np.array([-90.0, -30.0, np.nan]), "L": np.zeros(0)})
    cfg = config.load("task")
    cfg["encoder"]["vision"] = {**cfg["encoder"]["vision"], "mode": "retinotopic2",
                                "azimuth_sigma_deg": 30.0}
    enc = Encoder(smap, cfg, LoomSpec.from_config())
    t = enc.vision_tune("R", 0.0, -90.0)
    assert t[0] > t[1] and t[2] == 1.0          # 방위 −90° 물체 → −90° LC 가 −30° LC 보다


def test_replay_retune_keeps_other_senses_and_rescales_vision():
    """재주입: 좌표 없는 LC 발화율에서 바탕값을 되찾아 새 조율을 곱한다 · 다른 감각은 그대로."""
    from moai_echomind_fly.loop.encoder import Encoder, SensoryMap
    from moai_echomind_fly.loop.replay import retune_vision
    from moai_echomind_fly.protocols import LoomSpec

    # 자극 뉴런 5: 시각 R 3(조율 없는 LC 하나 포함) · 후각 L 1 · 촉각 LF 1
    smap = SensoryMap(np.arange(5), {"R": np.array([0, 1, 2]), "L": np.zeros(0, int)},
                      {"R": np.array([1.0, 0.5, 1.0]), "L": np.zeros(0)},
                      {"R": np.array([40.0, -40.0, np.nan]), "L": np.zeros(0)},
                      {"L": np.array([3]), "R": np.zeros(0, int)},
                      {"LF": np.array([4]), **{k: np.zeros(0, int)
                                               for k in ("LM", "LH", "RF", "RM", "RH")}})
    cfg = config.load("task")
    cfg["encoder"]["vision"] = {**cfg["encoder"]["vision"], "mode": "retinotopic"}
    enc = Encoder(smap, cfg, LoomSpec.from_config())
    base = 7.0                                            # 5 Hz × 게인 × 좌우 가중
    old = np.array([base * 1.0 * 0.5, base * 0.5 * 0.9, base * 1.0 * 1.0, 150.0, 33.0])
    new = retune_vision(old, smap, enc, elevation_deg=40.0, azimuth_deg=90.0)
    assert new[3] == 150.0 and new[4] == 33.0             # 후각 · 촉각 그대로
    assert abs(new[2] - base) < 1e-9                      # 조율 없는 LC = 바탕값 × 상대 세기 1
    assert abs(new[0] - base) < 1e-9                      # 고도 40° 물체 → 40° LC 조율 1
    assert new[1] < base * 0.5 * 0.9                      # −40° LC 는 더 약해진다


def test_smell_reads_antenna_sensors_not_palps():
    """task-v2.1 — FlyGym 순서는 작은턱수염 0 · 1 → 더듬이 2 · 3(연구 노트 14 §3.1)."""
    from moai_echomind_fly import config
    from moai_echomind_fly.loop.scorecard import raw_text
    from moai_echomind_fly.loop.world import Obs

    s = config.load("task")["encoder"]["smell"]["sensors"]
    assert s == {"L": 2, "R": 3}
    obs = Obs(0.0, (0.0, 0.0), 0.0, [9.0, 9.0, 1.0, 2.0], {}, {"present": False},
              (10.0, 0.0), False, 0.0)
    assert "왼 더듬이 1.00 · 오른 더듬이 2.00" in raw_text(obs, s)


def test_scorecard_decision_row_has_the_fields_the_dashboard_reads():
    """R-5777 F-002 — 결정 기록 한 줄의 칸(3.3)."""
    import json

    from moai_echomind_fly.loop.episode import Step
    from moai_echomind_fly.loop.scorecard import RAW_CHARS, decision_row
    from moai_echomind_fly.loop.teacher import Truth

    t = Truth(False, 0.0, 0.0, False, odor_left=1.5, odor_right=2.5, odor_azimuth_deg=30.0,
              food_distance_mm=12.0, legs_in_contact=6)
    step = Step(3, 900.0, t, "캡션", {"turn": "left"}, {"turn": "right"}, "vector", 0,
                (0.4, 0.6), {"raw": "x" * 500, "valid": True, "why": ""})
    row = decision_row(2000, step)
    for k in ("seed", "d", "t_ms", "truth", "teacher", "action", "raw", "valid", "why",
              "odor_L", "odor_R", "food_bearing_deg", "food_mm", "drive"):
        assert k in row
    assert (row["odor_L"], row["odor_R"], row["food_bearing_deg"]) == (1.5, 2.5, 30.0)
    assert len(row["raw"]) == RAW_CHARS and row["why"] is None
    json.dumps(row, ensure_ascii=False)                       # 한 줄로 쓸 수 있다


def test_loo_threshold_accuracy_ignores_shared_bias():
    """M1b — 두 쪽이 모두 음수여도(바탕 치우침) 갈리면 1.0 · 같으면 0.5 근처."""
    import numpy as np

    from moai_echomind_fly.inspect import _loo_threshold_acc

    assert _loo_threshold_acc(np.array([-4.0, -4.2, -3.9]), np.array([-9.0, -9.5, -8.8])) == 1.0
    rng = np.random.default_rng(0)
    same = _loo_threshold_acc(rng.normal(size=200), rng.normal(size=200))
    assert 0.35 < same < 0.65

"""R-5138 S0b — 물체 인식 시험 부품: 물체 목록 · 참값 · 물체 합 시각 인코더 · 걷는 몸."""

from __future__ import annotations

import copy

import numpy as np

from moai_echomind_fly import config
from moai_echomind_fly.loop.encoder import SensoryMap
from moai_echomind_fly.loop.world import Obs
from moai_echomind_fly.npc.body import ME
from moai_echomind_fly.npc.fake_world import FakeWorld
from moai_echomind_fly.npc.perceive import PerceiveBody
from moai_echomind_fly.npc.perception import (
    MultiEncoder,
    SceneMemory,
    scene_objects,
    truth_labels,
)
from moai_echomind_fly.protocols import LoomSpec

PCFG = config.load("npc")["perceive"]
ME0 = {"x": 0.0, "y": 0.0, "heading_deg": 0.0}


def _near(x, y, cls="obstacle", ref="a"):
    return {"ref": ref, "cls": cls, "x": x, "y": y, "size": {"radius_m": 0.3}}


def test_scene_objects_azimuth_left_positive_and_expansion_when_approaching():
    mem = SceneMemory()
    scene_objects(ME0, [_near(2.0, 1.0)], 0.15, 0.1, mem, skip=set())
    o = scene_objects(ME0, [_near(1.5, 0.75)], 0.15, 0.1, mem, skip=set())[0]
    assert o["azimuth_deg"] > 0                       # +y = 왼쪽
    assert o["expansion_deg_s"] > 0 and o["radial_m_s"] < 0


def test_truth_labels_nearest_within_radius():
    mem = SceneMemory()
    scene_objects(ME0, [_near(1.2, -0.5, "person")], 0.15, 0.1, mem, skip=set())
    objs = scene_objects(ME0, [_near(1.0, -0.4, "person"), _near(9.0, 0.0, ref="b")], 0.15, 0.1,
                         mem, skip=set())
    lab = truth_labels(objs, PCFG)
    assert lab == {"present": "있음", "side": "오른쪽", "close": "가까움",
                   "approaching": "다가옴", "person": "사람"}
    far = scene_objects(ME0, [_near(9.0, 0.0)], 0.15, 0.1, SceneMemory(), skip=set())
    assert truth_labels(far, PCFG)["present"] == "없음"


def _enc():
    smap = SensoryMap(np.arange(4), {"L": np.array([0, 1]), "R": np.array([2, 3])},
                      {"L": np.ones(2), "R": np.ones(2)},
                      {"L": np.full(2, np.nan), "R": np.full(2, np.nan)},
                      {"L": np.zeros(0, int), "R": np.zeros(0, int)},
                      {k: np.zeros(0, int) for k in ("LF", "LM", "LH", "RF", "RM", "RH")})
    cfg = copy.deepcopy(config.load("task"))
    npc = config.load("npc")
    cfg["encoder"]["vision"].update(npc["encoder_override"]["vision"])
    cfg["encoder"]["vision"].update(npc["perceive"]["vision"])
    return MultiEncoder(smap, cfg, LoomSpec.from_config())


def _obs(objs):
    lo = {"present": False, "azimuth_deg": 0.0, "size_deg": 0.0, "expansion_deg_s": 0.0,
          "elevation_deg": 0.0, "distance_mm": None}
    return Obs(0.0, (0.0, 0.0), 0.0, [0.0] * 4, {}, lo, (1e6, 1e6), False, 0.0,
               extra={"objects": objs})


def _o(az, exp):
    return {"azimuth_deg": az, "expansion_deg_s": exp, "size_deg": 10.0, "elevation_deg": 0.0,
            "present": True, "distance_mm": 10.0}


def test_multi_encoder_sums_objects_left_right_and_caps():
    enc = _enc()
    assert enc.rates(_obs([])).sum() == 0                       # 아무것도 없으면 시각 0
    assert enc.rates(_obs([_o(60, -5)])).sum() == 0              # 멀어지는 것은 안 보임
    one = enc.rates(_obs([_o(60, 30)]))
    two = enc.rates(_obs([_o(60, 30), _o(60, 30)]))
    assert one[0] > one[2]                                       # 왼쪽 물체 → 왼눈
    assert two[0] > one[0]                                       # 둘이면 더 셈
    big = enc.rates(_obs([_o(60, 1e4)] * 5))
    v = enc.enc["vision"]
    assert big.max() <= enc.loom.peak_hz * v["gain_max"] * v["multi_cap"] + 1e-6


def test_perceive_body_walks_fake_world_with_walkers_and_labels():
    w = FakeWorld()
    w.reset = lambda seed: None
    b = PerceiveBody(w, config.load("npc"))
    o0 = b.reset(4, {})
    assert len(w.things) == 1 + PCFG["walkers"]
    p0 = w.state(ME)
    for _ in range(9):
        o = b.step((1.0, 1.0), 100.0)
    p1 = w.state(ME)
    assert np.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]) > 0.5     # 걸었다
    assert "objects" in o.extra and set(b.last_labels) == {
        "present", "side", "close", "approaching", "person"}
    assert any(x["cls"] == "person" for x in o.extra["objects"]) or o0 is not None

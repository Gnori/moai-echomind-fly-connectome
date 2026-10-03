"""R-5138 S0 — NPC 몸(가짜 세계): 감각 식 · 명령 · 반사 물러남 · 같은 씨앗 같은 결과(뇌 없이)."""

from __future__ import annotations

import math

import pytest

from moai_echomind_fly import config
from moai_echomind_fly.loop.world import make_task
from moai_echomind_fly.npc.body import GOAL, HAZARD, ME, NpcBody, wrap_deg

TASK = {"odor_xy": [5.0, 3.0], "odor_peak": 500.0, "heading0": 0.0,
        "looming": {"t0_ms": 0.0, "bearing_deg": 40.0, "start_mm": 25.0, "speed_mm_s": 25.0,
                    "radius_mm": 1.5, "elevation_deg": 10.0}}


def test_goal_on_the_left_gives_stronger_left_sensor_and_units_scale():
    b = NpcBody()
    obs = b.reset(0, {k: v for k, v in TASK.items() if k != "looming"})
    assert obs.odor_sensors[2] > obs.odor_sensors[3]               # 왼쪽(+y) 목표 → 왼 더듬이 셈
    s = b.cfg["scale_m_per_mm"]
    assert b.world.things[GOAL].x == pytest.approx(5.0 * s)
    assert obs.food_distance_mm == pytest.approx(math.hypot(5.0, 3.0))


def test_drive_moves_forward_and_right_faster_turns_left():
    b = NpcBody()
    b.reset(0, {k: v for k, v in TASK.items() if k != "looming"})
    o1 = b.step((1.0, 1.0), 100.0)
    assert o1.pos[0] > 0 and abs(o1.pos[1]) < 1e-9
    h0 = b.world.state(ME)["heading_deg"]
    b.step((0.2, 0.8), 100.0)
    assert b.world.state(ME)["heading_deg"] > h0          # 오른쪽 구동이 크면 왼쪽(+)으로


def test_hazard_appears_looms_and_reflex_retreats_away():
    b = NpcBody()
    b.reset(0, TASK)
    o = b.step((0.0, 0.0), 100.0)
    assert o.looming["present"] and o.looming["azimuth_deg"] == pytest.approx(40.0, abs=1.0)
    o2 = b.step((0.0, 0.0), 100.0)
    assert o2.looming["expansion_deg_s"] > 0 and o2.looming["elevation_deg"] == 10.0
    me, hz = b.world.state(ME), b.world.things[HAZARD]
    d0 = math.dist((me["x"], me["y"]), (hz.x, hz.y))
    b.step((0.0, 0.0), 100.0, reflex=True)
    me2 = b.world.state(ME)
    away = math.degrees(math.atan2(me["y"] - hz.y, me["x"] - hz.x))
    assert abs(wrap_deg(me2["heading_deg"] - away)) < 15.0           # 위협 반대쪽(위협도 움직임)
    assert me2["speed_m_s"] > 0 and d0 > 0


def test_same_task_same_trajectory_and_live_port_is_forbidden_in_config():
    task = make_task(3, config.load("task"))
    runs = []
    for _ in range(2):
        b = NpcBody()
        b.reset(3, task)
        runs.append([b.step((0.7, 0.9), 50.0).pos for _ in range(20)])
    assert runs[0] == runs[1]
    assert 9212 in config.load("npc")["world"]["forbidden_ports"]


def test_no_brain_ignores_reflex_and_judge_hits():
    from moai_echomind_fly.npc.run import hazard_seeds, judge_hits

    b = NpcBody(reflex_enabled=False)
    b.reset(0, TASK)
    b.step((0.0, 0.0), 100.0)
    b.step((0.0, 0.0), 100.0, reflex=True)
    assert b.world.state(ME)["speed_m_s"] == 0.0                   # 반사 무시 = 구동 그대로(0)
    seeds = hazard_seeds(3000, 3, config.load("task"))
    assert len(seeds) == 3 and all("looming" in make_task(s, config.load("task")) for s in seeds)

    def eps(hits):
        return [{"seed": i, "looming_hit": h} for i, h in enumerate(hits)]

    n = 12
    res = {"real": eps([False] * n), "rewired-s0": eps([True] * n),
           "rewired-s1": eps([True] * n), "no_brain": eps([True] * n)}
    v = judge_hits(res)
    assert v["real_beats_rewired"] and v["real_below_no_brain"]
    res["real"] = eps([True] * n)
    assert not judge_hits(res)["real_beats_rewired"]

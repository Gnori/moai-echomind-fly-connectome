"""NPC 몸(R-5138 I · S0) — `Loop` 의 몸 자리(FlyGym `BodyClient`)에 끼우는 같은 모양.

`reset(seed, task) → Obs` · `step(drive, bin_ms, reflex=False) → Obs` · `close()`.
과제(make_task · 초파리 mm 단위)를 세계(미터)에 옮겨 장면을 만들고, 감각은 FlyGym 과 같은 식
(냄새 = 봉우리 / 거리² · 다가옴 = 시각 크기 · 확대 속도 · 방위 + 왼쪽)으로 NPC 위치에서 계산한다.
명령: 구동(왼 · 오른) → 속도 · 회전 · 반사(도망 뉴런 발화)면 위협 반대쪽으로 물러남.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .. import config
from ..loop.world import Obs
from .fake_world import FakeWorld, Thing

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
ME, GOAL, HAZARD = "npc", "goal", "hazard"


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class NpcBody:
    wants_reflex = True                     # Loop 이 반사 여부를 넘겨 준다(FlyGym 몸은 안 받음)

    def __init__(self, cfg: dict[str, Any] | None = None, world=None,
                 reflex_enabled: bool = True) -> None:
        self.cfg = cfg or config.load("npc")
        self.reflex_enabled = reflex_enabled   # False = 대조 '뇌 없음'(반사 무시 · 그대로 걸음)
        self.s = float(self.cfg["scale_m_per_mm"])
        self.world = world
        self.task: dict[str, Any] = {}
        self.t_ms = 0.0
        self.prev_size: float | None = None
        self.hazard_spawned = False

    # ---------------------------------------------------------- 장면
    def reset(self, seed: int, task: dict[str, Any]) -> Obs:
        if self.world is None or isinstance(self.world, FakeWorld):
            self.world = FakeWorld()
        self.task, self.t_ms, self.prev_size, self.hazard_spawned = task, 0.0, None, False
        h0 = math.degrees(float(task.get("heading0", 0.0)))
        self.world.add(Thing(ME, "resident", 0.0, 0.0, heading_deg=h0))
        gx, gy = task["odor_xy"]
        self.world.add(Thing(GOAL, "object", gx * self.s, gy * self.s, radius_m=0.2,
                             height_m=0.5))
        return self._obs(0.0)

    def _spawn_hazard(self) -> None:
        lo = self.task["looming"]
        me = self.world.state(ME)
        b = math.radians(me["heading_deg"] + lo["bearing_deg"])
        start = lo["start_mm"] * self.s
        x, y = me["x"] + start * math.cos(b), me["y"] + start * math.sin(b)
        self.world.add(Thing(HAZARD, "resident", x, y, radius_m=lo["radius_mm"] * self.s,
                             speed_m_s=lo["speed_mm_s"] * self.s, target=(me["x"], me["y"])))
        self.hazard_spawned = True

    # ---------------------------------------------------------- 감각
    def _obs(self, dt_s: float) -> Obs:
        me = self.world.state(ME)
        x, y = me["x"] / self.s, me["y"] / self.s                 # 과제 단위(mm)
        h = math.radians(me["heading_deg"])
        sc = self.cfg["senses"]
        peak = float(sc["goal_peak"] or self.task.get("odor_peak", 500.0))
        gx, gy = self.task["odor_xy"]
        fx, fy = math.cos(h), math.sin(h)
        lx, ly = -fy, fx                                            # 왼쪽(+90°)
        f, a = float(sc["antenna_forward_mm"]), float(sc["antenna_half_mm"])

        def conc(sign: float) -> float:
            px, py = x + f * fx + sign * a * lx, y + f * fy + sign * a * ly
            return peak / max((gx - px) ** 2 + (gy - py) ** 2, 1e-6)

        cl, cr = conc(+1.0), conc(-1.0)
        looming = {"present": False, "azimuth_deg": 0.0, "size_deg": 0.0,
                   "expansion_deg_s": 0.0, "elevation_deg": 0.0, "distance_mm": None}
        near = me.get("blocked", False)
        for o in self.world.neighbors(ME, 1e9):
            ox, oy = o["x"] / self.s, o["y"] / self.s
            d = math.hypot(ox - x, oy - y)
            if o["ref"] != GOAL and d <= float(sc["contact_near_mm"]) + o["radius_m"] / self.s:
                near = True
            if o["ref"] != HAZARD:
                continue
            r = o["radius_m"] / self.s
            size = math.degrees(2.0 * math.atan2(r, max(d, 1e-6)))
            az = wrap_deg(math.degrees(math.atan2(oy - y, ox - x)) - me["heading_deg"])
            exp = 0.0 if self.prev_size is None or dt_s <= 0 else (size - self.prev_size) / dt_s
            self.prev_size = size
            looming = {"present": True, "azimuth_deg": az, "size_deg": size,
                       "expansion_deg_s": exp,
                       "elevation_deg": float(self.task["looming"].get("elevation_deg", 0.0)),
                       "distance_mm": d}
        force = float(sc["contact_force"]) if near else 0.0
        legs = {leg: (force if leg in ("LF", "RF") else 0.0) for leg in LEGS}
        return Obs(self.t_ms, (x, y), h, [cl, cr, cl, cr], legs, looming,
                   tuple(self.task["odor_xy"]), False, time.time())

    # ---------------------------------------------------------- 움직임
    def step(self, drive: tuple[float, float], bin_ms: float, reflex: bool = False) -> Obs:
        dt = bin_ms / 1000.0
        lo = self.task.get("looming")
        if lo and not self.hazard_spawned and self.t_ms >= lo["t0_ms"]:
            self._spawn_hazard()
        if self.hazard_spawned and self.cfg.get("s1", {}).get("hazard_pursue"):
            me0 = self.world.state(ME)            # 위험이 지금 자리를 쫓는다(S1 장면)
            if HAZARD in getattr(self.world, "things", {}):
                self.world.things[HAZARD].target = (me0["x"], me0["y"])
        me = self.world.state(ME)
        m = self.cfg["motor"]
        heading, speed_mm = me["heading_deg"], 0.0
        threat = next((o for o in self.world.neighbors(ME, 1e9) if o["ref"] == HAZARD), None)
        if reflex and self.reflex_enabled and threat is not None:
            away = math.degrees(math.atan2(me["y"] - threat["y"], me["x"] - threat["x"]))
            if self.cfg.get("reflex", {}).get("mode", "retreat") == "sidestep":
                # 위협이 다가오는 길에서 옆으로 — 위협이 왼쪽(+)이면 오른쪽(−90°)으로
                toward = wrap_deg(away + 180.0 - me["heading_deg"])
                away = wrap_deg(me["heading_deg"] + (-90.0 if toward >= 0 else 90.0))
            heading, speed_mm = away, float(m["reflex_retreat_mm_s"])
        else:
            left, right = drive
            speed_mm = max(0.0, min(1.5, 0.5 * (left + right))) * float(m["max_speed_mm_s"])
            heading = wrap_deg(heading + float(m["turn_gain_deg_s"]) * (right - left) * dt)
        self.world.steer(ME, heading, speed_mm * self.s, dt)
        self.world.step(dt)
        self.t_ms += bin_ms
        return self._obs(dt)

    def close(self) -> None:
        pass

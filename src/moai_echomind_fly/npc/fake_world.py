"""시험용 가짜 세계(R-5138 S0) — 2D 점 · 원 · 직선 이동. 상자 월드와 같은 메서드 모양.

좌표는 미터 · 방향은 세계 기준 도(+ = 반시계 · x 축 0°). 한 걸음씩 진행(lockstep): `steer` 로 명령을
걸고 `step(dt)` 만큼만 세계가 움직인다(다른 개체 포함).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Thing:
    ref: str
    kind: str                       # resident | object
    x: float
    y: float
    radius_m: float = 0.25
    height_m: float = 1.7
    heading_deg: float = 0.0
    speed_m_s: float = 0.0
    target: tuple[float, float] | None = None   # 있으면 그쪽으로 곧게(위험이 다가오기)


@dataclass
class FakeWorld:
    things: dict[str, Thing] = field(default_factory=dict)
    t_s: float = 0.0
    blocked: dict[str, bool] = field(default_factory=dict)

    def add(self, thing: Thing) -> str:
        self.things[thing.ref] = thing
        return thing.ref

    def remove(self, ref: str) -> None:
        self.things.pop(ref, None)

    def place(self, ref: str, x: float, y: float, heading_deg: float = 0.0) -> dict:
        t = self.things[ref]
        t.x, t.y, t.heading_deg = x, y, heading_deg
        return self.state(ref)

    def state(self, ref: str) -> dict[str, Any]:
        t = self.things[ref]
        return {"ref": ref, "x": t.x, "y": t.y, "heading_deg": t.heading_deg,
                "speed_m_s": t.speed_m_s, "blocked": self.blocked.get(ref, False), "t_s": self.t_s}

    def neighbors(self, ref: str, radius_m: float) -> list[dict[str, Any]]:
        me = self.things[ref]
        out = []
        for o in self.things.values():
            if o.ref != ref and math.dist((me.x, me.y), (o.x, o.y)) <= radius_m:
                out.append({"ref": o.ref, "kind": o.kind, "x": o.x, "y": o.y,
                            "radius_m": o.radius_m, "height_m": o.height_m,
                            "heading_deg": o.heading_deg, "speed_m_s": o.speed_m_s})
        return out

    def steer(self, ref: str, heading_deg: float, speed_m_s: float, dt_s: float) -> dict:
        t = self.things[ref]
        t.heading_deg, t.speed_m_s = heading_deg, max(0.0, speed_m_s)
        return self.state(ref)

    def step(self, dt_s: float) -> float:
        for t in self.things.values():
            if t.target is not None and t.speed_m_s > 0:
                dx, dy = t.target[0] - t.x, t.target[1] - t.y
                t.heading_deg = math.degrees(math.atan2(dy, dx))
            h = math.radians(t.heading_deg)
            t.x += t.speed_m_s * math.cos(h) * dt_s
            t.y += t.speed_m_s * math.sin(h) * dt_s
        self.t_s += dt_s
        return self.t_s

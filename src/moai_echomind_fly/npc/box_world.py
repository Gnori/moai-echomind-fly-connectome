"""상자 월드 클라이언트(R-5138 S0) — `FakeWorld` 와 같은 메서드 모양으로 실험 입구에 붙는다.

입구(echomind-world 10-03 · 브랜치 fly-steer-1003 · 문서 experiment-inputs.md):
`/exp/reset` · `/exp/bodies` · `/exp/steer/{id}` · `/exp/step` · `/exp/near/{id}` ·
`/exp/body/{id}`. 상자 월드만(라이브 9212 거부).

좌표 대응: 우리 x = 세계 z · 우리 y = 세계 x · 방향은 같은 수(세계 0° = +z · 90° = +x ·
왼쪽으로 커짐 = 우리 + = 왼쪽). 위험 몸은 우리가 매 진행 전에 목표 쪽으로 조종한다
(상자 월드에선 주민 · 사물을 못 만듦). 부딪힘: 그쪽 몸은 겹치지 않고 반경 둘 앞에서 멈춘다
→ 위험 몸의 조종이 NPC 에 막히거나 접촉이면 `hit`.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .fake_world import Thing


class BoxWorld:
    def __init__(self, base_url: str, forbidden_ports: list[int], timeout_s: float = 30.0,
                 opener=None, origin: tuple[float, float] = (0.0, 0.0)) -> None:
        port = urlparse(base_url).port
        if port in set(forbidden_ports):
            raise ValueError(f"포트 {port} 는 라이브 월드 — 붙지 않는다(config/npc.yaml)")
        self.base = base_url.rstrip("/")
        self.origin = (float(origin[0]), float(origin[1]))   # 우리 (0, 0) 의 세계 (x, z) — 빈 땅
        self.timeout = timeout_s
        self.open = opener or urllib.request.urlopen
        self.things: dict[str, Thing] = {}           # 우리가 조종하는 몸(목표 · 위험 · NPC)
        self.virtual: set[str] = set()               # 세계에 없는 것(목표 — 사물을 못 만듦)
        self.last: dict[str, dict] = {}
        self.hit = False
        self.t_s = 0.0

    # -------------------------------------------------------- HTTP
    def _req(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"content-type": "application/json"})
        with self.open(req, timeout=self.timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}

    # -------------------------------------------------------- 좌표
    def to_world(self, x: float, y: float) -> tuple[float, float]:
        return y + self.origin[0], x + self.origin[1]          # (세계 x, 세계 z)

    def from_world(self, wx: float, wz: float) -> tuple[float, float]:
        return wz - self.origin[1], wx - self.origin[0]

    # -------------------------------------------------------- 장면
    def reset(self, seed: int) -> None:
        self._req("POST", "/exp/reset", {"seed": int(seed)})
        self.things, self.virtual, self.last, self.hit, self.t_s = {}, set(), {}, False, 0.0

    def add(self, thing: Thing) -> str:
        self.things[thing.ref] = thing
        if thing.kind == "object":                    # 목표는 세계에 없다(감각은 과제 좌표로)
            self.virtual.add(thing.ref)
            return thing.ref
        wx, wz = self.to_world(thing.x, thing.y)
        self._req("POST", "/exp/bodies", {"id": thing.ref, "x": wx, "z": wz,
                                          "heading_deg": thing.heading_deg,
                                          "height_m": thing.height_m})
        return thing.ref

    def remove(self, ref: str) -> None:
        self.things.pop(ref, None)
        if ref not in self.virtual:
            self._req("DELETE", f"/exp/bodies/{ref}")

    def body_rules(self, ref: str) -> dict[str, Any]:
        return self._req("GET", f"/exp/body/{ref}")

    # -------------------------------------------------------- 읽기
    def state(self, ref: str) -> dict[str, Any]:
        n = self._req("GET", f"/exp/near/{ref}?r=0.01")["self"]
        x, y = self.from_world(n["x"], n["z"])
        return {"ref": ref, "x": x, "y": y, "heading_deg": n["heading_deg"],
                "speed_m_s": n.get("speed", 0.0),
                "blocked": bool(self.last.get(ref, {}).get("blocked", False)), "t_s": self.t_s}

    def neighbors(self, ref: str, radius_m: float) -> list[dict[str, Any]]:
        r = min(float(radius_m), 200.0)
        out = []
        for o in self._req("GET", f"/exp/near/{ref}?r={r}")["near"]:
            x, y = self.from_world(o["x"], o["z"])
            size = o.get("size") or {}
            out.append({"ref": o["id"], "kind": o.get("class"), "x": x, "y": y,
                        "radius_m": float(size.get("radius_m") or 0.3),
                        "height_m": float(size.get("height_m") or 1.7),
                        "heading_deg": o.get("heading_deg", 0.0),
                        "speed_m_s": o.get("speed", 0.0)})
        for ref2 in self.virtual:                     # 목표(가상)도 이웃으로
            t = self.things[ref2]
            out.append({"ref": t.ref, "kind": "object", "x": t.x, "y": t.y,
                        "radius_m": t.radius_m, "height_m": t.height_m,
                        "heading_deg": 0.0, "speed_m_s": 0.0})
        return out

    def neural(self, ref: str, payload: dict[str, Any]) -> None:
        """신경 상태 띠(보기용 · 판정에 안 씀) — 입구가 없으면 조용히 넘어간다."""
        try:
            self._req("POST", f"/exp/neural/{ref}", payload)
        except Exception:                             # 보기용 — 실험을 멈추지 않는다
            pass

    # -------------------------------------------------------- 움직임
    def steer(self, ref: str, heading_deg: float, speed_m_s: float, dt_s: float) -> dict:
        got = self._req("POST", f"/exp/steer/{ref}",
                        {"heading_deg": heading_deg % 360.0, "speed": max(0.0, speed_m_s),
                         "dt_s": dt_s})
        self.last[ref] = got
        return got

    def step(self, dt_s: float) -> float:
        for t in self.things.values():                # 위험 몸 — 목표 쪽으로 우리가 조종
            if t.ref in self.virtual or t.target is None or t.speed_m_s <= 0:
                continue
            me = self.state(t.ref)
            h = math.degrees(math.atan2(t.target[1] - me["y"], t.target[0] - me["x"]))
            got = self.steer(t.ref, h, t.speed_m_s, dt_s)
            if got.get("blocked") and (got.get("blocked_by") or {}).get("id") in self.things:
                self.hit = True
            if any(c.get("id") in self.things and c.get("id") != t.ref
                   for c in got.get("contacts") or []):
                self.hit = True
        got = self._req("POST", "/exp/step", {"dt_s": dt_s})
        for ref, cs in (got.get("contacts") or {}).items():
            if ref in self.things and any(c.get("id") in self.things for c in cs):
                self.hit = True
        self.t_s = float(got.get("sim_s", self.t_s + dt_s))
        return self.t_s

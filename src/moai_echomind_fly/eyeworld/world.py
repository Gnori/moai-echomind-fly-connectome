"""세계 서버 — 세계 상태의 정본(설계 문서 I 2.7). 렌더링 없음 · 같은 씨앗이면 같은 궤적.

좌표(에코월드와 같음): 미터 · y 위 · 오른손 · yaw 0 = +z 쪽을 봄 · yaw 가 늘면 **왼쪽**으로 돈다.
  앞 f = (sin yaw, 0, cos yaw) · 왼 l = (cos yaw, 0, −sin yaw) · 위 u = (0, 1, 0).
몸 방향(초파리 실험 규약): 방위 + = 왼쪽 · 0 = 정면 · 고도 + = 위.
  방향 벡터 d = cos(el)·cos(az)·f + cos(el)·sin(az)·l + sin(el)·u.
눈: 머리(눈높이 = 키 × eye_frac) 좌우로 ±eye_sep/2 · 눈마다 큐브맵 6 면(몸 축에 맞춤).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def forward(yaw: float) -> np.ndarray:
    return np.array([np.sin(yaw), 0.0, np.cos(yaw)])


def left(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw), 0.0, -np.sin(yaw)])


def body_dir(yaw: float, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """몸 기준 (방위, 고도) → 세계 방향 단위 벡터(…, 3)."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    f, lv = forward(yaw), left(yaw)
    ca = np.cos(e)[..., None]
    return (
        ca * np.cos(a)[..., None] * f + ca * np.sin(a)[..., None] * lv + np.sin(e)[..., None] * UP
    )


@dataclass
class Body:
    """몸 규약 — 에코월드 상자 실험 판과 같은 값(설정 eyeworld.yaml world.body)."""

    radius_m: float = 0.35
    height_m: float = 1.6
    eye_frac: float = 0.936
    eye_sep_m: float = 0.07
    walk_max_m_s: float = 1.3
    run_max_m_s: float = 4.5
    turn_max_deg_s: float = 180.0


@dataclass
class Command:
    """몸 명령 — 앞 속도(m/s) · 회전(°/s · + 왼쪽) · 옆걸음(m/s · + 왼쪽)."""

    speed: float = 0.0
    turn: float = 0.0
    side: float = 0.0


@dataclass
class Entity:
    id: str
    kind: str  # agent | person | animal | prop
    asset: str | None
    pos: np.ndarray  # (x, y, z) m · y = 발 높이
    yaw: float = 0.0
    radius: float = 0.3
    height: float = 1.0
    scale: float = 1.0
    solid: bool = True
    script: dict | None = None
    anim: dict | None = None
    body: Body | None = None
    col_offset: tuple[float, float] = (
        0.0,
        0.0,
    )  # 충돌 원기둥 중심의 모형 원점 대비 (x, z) — 개체 yaw 로 돌림
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def col_center(self) -> np.ndarray:
        ox, oz = self.col_offset
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return self.pos + np.array([c * ox + s * oz, 0.0, -s * ox + c * oz])


class World:
    """개체 · 몸 · 충돌 · 대본 · 시계."""

    def __init__(self, entities: list[Entity], seed: int = 0) -> None:
        self.t = 0.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ents = {e.id: e for e in entities}
        self.contacts: list[tuple[str, str]] = []

    def agent(self, eid: str) -> Entity:
        e = self.ents[eid]
        if e.body is None:
            raise KeyError(f"{eid} 는 몸이 없다")
        return e

    def _script(self, e: Entity, dt: float) -> None:
        s = e.script or {}
        kind = s.get("kind")
        if kind is None or self.t < float(s.get("start_s", 0.0)):
            e.vel = np.zeros(3)
            return
        if kind == "approach":  # 목표 개체의 시작 자리(또는 지금 자리)로 곧게
            tgt = self.ents[s["target"]].pos if s.get("pursue") else np.asarray(s["_aim"], float)
            d = tgt - e.pos
            d[1] = 0.0
            n = float(np.linalg.norm(d))
            v = float(s["speed"])
            e.vel = d / n * v if n > 1e-9 else np.zeros(3)
            if n > 1e-9:
                e.yaw = float(np.arctan2(d[0], d[2]))
        elif kind == "path":  # 점들을 차례로 · 끝나면 멈춤
            pts = s["points"]
            k = int(s.get("_k", 0))
            if k >= len(pts):
                e.vel = np.zeros(3)
                return
            d = np.array([pts[k][0], e.pos[1], pts[k][1]]) - e.pos
            n = float(np.linalg.norm(d))
            v = float(s["speed"])
            if n <= v * dt:
                s["_k"] = k + 1
            e.vel = d / n * v if n > 1e-9 else np.zeros(3)
            if n > 1e-9:
                e.yaw = float(np.arctan2(d[0], d[2]))
        else:
            raise ValueError(f"모르는 대본 {kind}")
        e.pos = e.pos + e.vel * dt

    def step(self, dt: float, commands: dict[str, Command] | None = None) -> list[tuple[str, str]]:
        """dt 초 진행 — 대본 → 몸 명령(상한) → 충돌(밀어냄) → 닿음 목록."""
        commands = commands or {}
        for e in self.ents.values():
            if (
                e.script is not None
                and e.script.get("kind") == "approach"
                and "_aim" not in e.script
            ):
                e.script["_aim"] = self.ents[e.script["target"]].pos.copy()
        for e in self.ents.values():
            if e.script is not None:
                self._script(e, dt)
        for eid, c in commands.items():
            e = self.agent(eid)
            b = e.body
            assert b is not None
            turn = float(np.clip(c.turn, -b.turn_max_deg_s, b.turn_max_deg_s))
            e.yaw = float(e.yaw + np.radians(turn) * dt)
            sp = float(np.clip(c.speed, -b.walk_max_m_s, b.run_max_m_s))
            sd = float(np.clip(c.side, -b.run_max_m_s, b.run_max_m_s))
            e.vel = forward(e.yaw) * sp + left(e.yaw) * sd
            e.pos = e.pos + e.vel * dt
        self.contacts = []
        ids = sorted(self.ents)
        for i, a in enumerate(ids):
            for b_id in ids[i + 1 :]:
                ea, eb = self.ents[a], self.ents[b_id]
                d = eb.col_center() - ea.col_center()
                d[1] = 0.0
                n = float(np.linalg.norm(d))
                gap = ea.radius + eb.radius - n
                if gap <= 0:
                    continue
                self.contacts.append((a, b_id))
                if not (ea.solid and eb.solid):
                    continue
                u = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
                if ea.body is not None and eb.body is None:  # 몸만 밀려남
                    ea.pos = ea.pos - u * gap
                elif eb.body is not None and ea.body is None:
                    eb.pos = eb.pos + u * gap
                else:
                    ea.pos, eb.pos = ea.pos - u * gap / 2, eb.pos + u * gap / 2
        self.t += dt
        return list(self.contacts)

    def poses(self) -> list[dict]:
        """렌더러에 보낼 자세 묶음(설계 문서 2.6 render.poses)."""
        out = []
        for eid in sorted(self.ents):
            e = self.ents[eid]
            if e.asset is None:
                continue
            out.append(
                {
                    "id": eid,
                    "asset": e.asset,
                    "pos": [round(float(x), 6) for x in e.pos],
                    "yaw": round(float(e.yaw), 6),
                    "scale": e.scale,
                    **({"anim": e.anim} if e.anim else {}),
                }
            )
        return out

    def eye_rig(self, eid: str) -> list[dict]:
        """눈 둘의 자리와 큐브맵 6 면(앞 · 뒤 · 왼 · 오 · 위 · 아래)의 forward · up(세계 좌표)."""
        e = self.agent(eid)
        b = e.body
        assert b is not None
        f, lv = forward(e.yaw), left(e.yaw)
        head = e.pos + UP * b.height_m * b.eye_frac
        faces = [
            ("front", f, UP),
            ("back", -f, UP),
            ("left", lv, UP),
            ("right", -lv, UP),
            ("up", UP, -f),
            ("down", -UP, f),
        ]
        rig = []
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            pos = head + lv * sgn * b.eye_sep_m / 2
            rig.append(
                {
                    "agent": eid,
                    "side": side,
                    "pos": [float(x) for x in pos],
                    "faces": [
                        {"name": n, "forward": [float(x) for x in fw], "up": [float(x) for x in up]}
                        for n, fw, up in faces
                    ],
                }
            )
        return rig

    def state_hash(self) -> str:
        s = json.dumps(
            [
                (k, [round(float(x), 9) for x in e.pos], round(float(e.yaw), 9))
                for k, e in sorted(self.ents.items())
            ]
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]

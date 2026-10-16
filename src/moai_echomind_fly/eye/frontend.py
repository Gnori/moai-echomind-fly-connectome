"""시각 앞단 v2 — 그림(눈마다 큐브맵) → T4/T5 · 소엽 기둥 세포 발화율.

설계 문서 I 3.3b · P eye Task 2.
기둥: L1 공식 좌표(눈마다 875 · 892) → (h, v) → 방위 · 고도(retino-v2 선언)
  → 몸 기준 방향 → 큐브맵 표본.
T4(ON) · T5(OFF) 국소 움직임 = 옆쌍 상관 검출기(Zhou 2022 · τ 30 ms · 노트 17 §6.2) — Hrc 참고.
  ref = 같은 검출기에 대비 1 사인 줄무늬(λ)를 v_ref 로 흘렸을 때의 평균(앞단만 · 뇌와 무관).
  (Haag 2016 세 갈래 · 250 ms 는 10-17 앞단 시험에서 DSI 0.05~0.37 — 쓰지 않음 · config 기록.)
방향(눈 격자 축): a 뒤로(−h) · b 앞으로(+h) · c 위(+v) · d 아래(−v).
  가로 이웃 Δh = ±2 · 세로 이웃 = 대각 둘 평균.
소엽 기둥 세포(Tm2 · Tm4 · T2 · Tm3): 뉴런 자리에서 수용장 σ 로 흐린 밝기의 dL/dt(V0b 와 같은 식).
⚠️ T4/T5 는 문헌상 발화하지 않는다(등급 전위) — 포아송 발화는 우리 약속.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config, retino
from ..eyeworld.world import Body, Entity, World, body_dir
from ..graph import Graph
from ..loop.signals import side_of
from . import disk
from .compound import Sampler
from .geometry import Positions, _linear, t4t5_positions, type_positions


def body_faces() -> list[dict]:
    """몸 축(yaw 0)에 맞춘 큐브맵 6 면 — 몸 기준 방향 = 세계 방향(yaw 0)."""
    w = World([Entity(id="a", kind="agent", asset=None, pos=np.zeros(3), body=Body())])
    return w.eye_rig("a")[0]["faces"]


@dataclass
class Columns:
    side: str
    hx: np.ndarray  # [C, 2] (hex1, hex2)
    h: np.ndarray
    v: np.ndarray
    az: np.ndarray
    el: np.ndarray


def eye_columns(g: Graph) -> dict[str, Columns]:
    """눈마다 기둥(L1 공식 좌표 · 고유) → 방위 · 고도(retino-v2 와 같은 범위 · 같은 식)."""
    rcfg = config.load("retino")
    typ = g.columns["type"].astype(str)
    side = side_of(g)
    hexes = retino.column_hex(g)
    cols = hexes[~np.isnan(hexes[:, 0])]
    hs_all, vs_all = cols[:, 0] - cols[:, 1], cols.sum(1)
    out = {}
    for s in ("L", "R"):
        m = (typ == "L1") & (side == s) & ~np.isnan(hexes[:, 0])
        hx = np.unique(hexes[m], axis=0)
        h, v = hx[:, 0] - hx[:, 1], hx.sum(1)
        az = _linear(
            h, float(hs_all.min()), float(hs_all.max()), *map(float, rcfg["azimuth_range_deg"])
        )
        az = -az if s == "L" else az
        el = _linear(
            v, float(vs_all.min()), float(vs_all.max()), *map(float, rcfg["elevation_range_deg"])
        )
        out[s] = Columns(s, hx, h, v, az, el)
    return out


def neighbours(c: Columns) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """방향마다 (위치 −1 이웃들, 위치 +1 이웃들) — 각 [C, 2] 기둥 번호(없으면 −1)."""
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(c.hx)}

    def look(d1: int, d2: int) -> np.ndarray:
        return np.array([key.get((int(a) + d1, int(b) + d2), -1) for a, b in c.hx])

    ant = look(1, -1)  # h + 2(앞)
    post = look(-1, 1)  # h − 2(뒤)
    up = np.stack([look(1, 0), look(0, 1)], 1)  # v + 1 둘
    down = np.stack([look(-1, 0), look(0, -1)], 1)  # v − 1 둘
    two = lambda x: np.stack([x, x], 1)  # noqa: E731
    return {
        "a": (two(ant), two(post)),
        "b": (two(post), two(ant)),
        "c": (down, up),
        "d": (up, down),
    }


class Hrc:
    """옆쌍 상관 검출기(기둥 × 4 방향 × ON/OFF) — 상태를 가진다.

    설정 frontend.t4t5 · model hrc-adjacent.

    입력 x = 정류(고역(L, τ_hp)) — ON 은 양의 쪽 · OFF 는 음의 쪽. 저역 l = LP(x, τ_lp).
    선호 방향(−1 → +1)의 출력 = ½[(l₋₁·x₀ − x₋₁·l₀) + (l₀·x₊₁ − x₀·l₊₁)] → 0 에서 자름.
    바로 옆 기둥끼리만 곱한다 — 앞 · 뒤 이웃을 바로 곱하면 17.6° 떨어져
    30° 줄무늬에서 방향이 뒤집힌다(10-17 시험).
    """

    def __init__(self, n_cols: int, nb: dict, dt_ms: float, p: dict) -> None:
        self.nb = nb
        self.a_hp = float(np.exp(-dt_ms / float(p["tau_hp_ms"])))
        self.a_lp = float(np.exp(-dt_ms / float(p["tau_lp_ms"])))
        self.prev = None
        self.hp = np.zeros(n_cols)
        self.lp = {pol: np.zeros(n_cols) for pol in ("on", "off")}

    def step(self, lum: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        if self.prev is None:
            self.prev = lum.copy()
        self.hp = self.a_hp * (self.hp + lum - self.prev)  # 1 차 고역
        self.prev = lum.copy()
        out = {}
        for pol, sgn in (("on", 1.0), ("off", -1.0)):
            x = np.maximum(sgn * self.hp, 0.0)
            self.lp[pol] = self.a_lp * self.lp[pol] + (1 - self.a_lp) * x
            lp = self.lp[pol]
            xx, ll = np.concatenate([x, [0.0]]), np.concatenate([lp, [0.0]])
            r = {}
            for d, (e_idx, s_idx) in self.nb.items():
                ok = (e_idx >= 0).all(1) & (s_idx >= 0).all(1)
                v1 = ll[e_idx].mean(1) * x - xx[e_idx].mean(1) * lp
                v2 = lp * xx[s_idx].mean(1) - x * ll[s_idx].mean(1)
                r[d] = np.where(ok, np.maximum((v1 + v2) / 2, 0.0), 0.0)
            out[pol] = r
        return out


def calibrate_ref(
    p: dict,
    spacing_deg: float,
    dt_ms: float,
    v_ref: float,
    lam_deg: float,
    n: int = 64,
    t_ms: float = 2000.0,
) -> float:
    """대비 1 사인 줄무늬를 v_ref 로 선호 방향(+)으로 흘린 1 차원 기둥 줄에서 출력 평균(뒤 절반)."""
    pos = np.arange(n) * spacing_deg
    idx = np.arange(n)
    e_idx = np.stack([np.where(idx - 1 >= 0, idx - 1, -1)] * 2, 1)
    s_idx = np.stack([np.where(idx + 1 < n, idx + 1, -1)] * 2, 1)
    det = Hrc(n, {"a": (e_idx, s_idx)}, dt_ms, p)
    vals = []
    steps = int(t_ms / dt_ms)
    for k in range(steps):
        lum = 0.5 + 0.5 * np.sin(2 * np.pi * (pos - v_ref * k * dt_ms / 1000.0) / lam_deg)
        r = det.step(lum)["on"]["a"]
        if k >= steps // 2:
            vals.append(r[2:-2].mean())
    return float(np.mean(vals))


class FrontendV2:
    """그림 → 입구 뉴런 발화율. `entry` 순서대로 [T4/T5 …, 소엽 기둥 세포 …]."""

    def __init__(self, g: Graph, face_px: int, dt_ms: float) -> None:
        cfg = config.load("eyeworld")["frontend"]
        self.cfg = cfg
        self.dt_ms = dt_ms
        self.r_max = float(cfg["r_max_hz"])
        faces = body_faces()
        self.cols = eye_columns(g)
        self.samplers, self.det = {}, {}
        for s, c in self.cols.items():
            dirs = body_dir(0.0, c.az, c.el)
            self.samplers[s] = Sampler(faces, face_px, float(cfg["acceptance_sigma_deg"])).fit(dirs)
            self.det[s] = Hrc(len(c.h), neighbours(c), dt_ms, cfg["t4t5"])
        sp = self.spacing_deg()
        self.ref = calibrate_ref(
            cfg["t4t5"],
            sp,
            dt_ms,
            float(cfg["t4t5"]["v_ref_deg_s"]),
            float(cfg["t4t5"]["lambda_deg"]),
        )
        self.t45 = t4t5_positions(g)
        self.t45_col = np.zeros(len(self.t45), dtype=np.int64)
        for s, c in self.cols.items():
            m = self.t45.side == s
            d2 = (self.t45.h[m, None] - c.h[None, :]) ** 2 + (
                self.t45.v[m, None] - c.v[None, :]
            ) ** 2
            self.t45_col[m] = d2.argmin(1)
        self.groups = {
            s: {
                (("on" if k == "T4" else "off"), sub): np.flatnonzero(
                    (self.t45.side == s) & (self.t45.kind == k) & (self.t45.sub == sub)
                )
                for k in ("T4", "T5")
                for sub in "abcd"
            }
            for s in ("L", "R")
        }
        lob_types = list(cfg["lobula"].keys())
        self.lob: Positions = type_positions(g, lob_types)
        self.lob_samplers = {}
        for t in lob_types:
            for s in ("L", "R"):
                m = (self.lob.kind == t) & (self.lob.side == s)
                if m.any():
                    sig = float(cfg["lobula"][t]["rf_sigma_deg"])
                    dirs = body_dir(0.0, self.lob.az[m], self.lob.el[m])
                    self.lob_samplers[(t, s)] = (
                        np.flatnonzero(m),
                        Sampler(faces, face_px, sig).fit(dirs),
                    )
        self.lob_prev = None
        self.lob_filt = np.zeros(len(self.lob))
        self.entry = np.concatenate([self.t45.idx, self.lob.idx])

    def spacing_deg(self) -> float:
        """가로 이웃 기둥(Δh = 2) 사이 각의 중앙값 — 기둥 배치에서 계산(손으로 정하지 않음)."""
        angs = []
        for c in self.cols.values():
            d = body_dir(0.0, c.az, c.el)
            ant = neighbours(c)["b"][1][:, 0]
            ok = ant >= 0
            angs.append(np.degrees(np.arccos(np.clip((d[ok] * d[ant[ok]]).sum(1), -1, 1))))
        return float(np.median(np.concatenate(angs)))

    def step(self, img: np.ndarray) -> np.ndarray:
        """img [2 눈(L, R), 6, N, N] u8 → 발화율 Hz(entry 순서)."""
        eyes = {"L": img[0], "R": img[1]}
        r45 = np.zeros(len(self.t45))
        for s in ("L", "R"):
            out = self.det[s].step(self.samplers[s].sample(eyes[s]))
            for (pol, sub), ix in self.groups[s].items():
                r45[ix] = out[pol][sub][self.t45_col[ix]]
        r45 = self.r_max * np.clip(r45 / self.ref, 0.0, 1.0)
        lum = np.zeros(len(self.lob))
        for (_, s), (ix, smp) in self.lob_samplers.items():
            lum[ix] = smp.sample(eyes[s])
        if self.lob_prev is None:
            self.lob_prev = lum.copy()
        x = (lum - self.lob_prev) / (self.dt_ms / 1000.0)
        self.lob_prev = lum
        rl = np.zeros(len(self.lob))
        v_ref = float(self.cfg["t4t5"]["v_ref_deg_s"])
        for t, p in self.cfg["lobula"].items():
            m = self.lob.kind == t
            resp = {"on": np.maximum(x[m], 0), "off": np.maximum(-x[m], 0), "onoff": np.abs(x[m])}[
                p["polarity"]
            ]
            a = float(np.exp(-self.dt_ms / float(p["tau_ms"])))
            self.lob_filt[m] = a * self.lob_filt[m] + (1 - a) * resp
            ref = disk.edge_ref(1.0, v_ref, float(p["rf_sigma_deg"]))
            rl[m] = self.r_max * np.clip(self.lob_filt[m] / ref, 0.0, 1.0)
        return np.concatenate([r45, rl])

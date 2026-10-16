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

앞단 v3(FrontendV3 · 노트 18 §6.2 · 설정 frontend_v3) = v2 + 두 단계(상관 앞):
  (가) 밝기 → 국소 웨버 대비 c = (L − Ī)/Ī · Ī = 같은 눈 기둥 밝기의 가우시안(FWHM 25°) 평균
       — 소엽 기둥 세포의 dL/dt 도 이 c 로.
  (나) 정류 뒤 · 곱하기 앞 나눗셈 정규화 r = x^p / (c50^p + x^p + (w·P)^p)
       · P = 같은 극성 x 의 가우시안(FWHM 30°) 무게 평균(Drews 2020).
  값은 손으로 고른 선언(문헌 범위 안) — 결과 전 잠금.
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


def gauss_weights(src: np.ndarray, dst: np.ndarray, fwhm_deg: float) -> np.ndarray:
    """방향 src [M, 3] 마다 방향 dst [C, 3] 의 가우시안(FWHM) 무게 — 행 합 1."""
    ang = np.arccos(np.clip(src @ dst.T, -1.0, 1.0))
    sig = np.radians(fwhm_deg) / 2.355
    w = np.exp(-0.5 * (ang / sig) ** 2)
    return w / w.sum(axis=1, keepdims=True)


class Weber:
    """밝기 → 국소 웨버 대비 c = (L − Ī)/Ī · Ī = 무게 w @ 둘레 밝기(노트 18 M1 · C1).

    Ī 가 floor 아래면 floor 로 나눈다(나눗셈 보호 · 선언).
    """

    def __init__(self, w: np.ndarray, floor: float) -> None:
        self.w = w
        self.floor = floor

    def __call__(self, lum: np.ndarray, around: np.ndarray | None = None) -> np.ndarray:
        mean = np.maximum(self.w @ (lum if around is None else around), self.floor)
        return (lum - mean) / mean


class DivNorm:
    """정류 뒤 · 곱하기 앞 나눗셈 정규화(Drews 2020 D1 · 노트 18 §6.2).

    r = x^p / (c50^p + x^p + (w_pool · P)^p) · P = pool @ x(행 합 1 인 가우시안 무게).
    정지 배경은 고역에서 사라지므로 P 는 움직이는 둘레만 담는다.
    """

    def __init__(self, pool: np.ndarray, p: float, w_pool: float, c50: float) -> None:
        self.pool = pool
        self.p = p
        self.w_pool = w_pool
        self.c50 = c50

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xp = x**self.p
        return xp / (self.c50**self.p + xp + (self.w_pool * (self.pool @ x)) ** self.p)


def line(n: int, spacing_deg: float) -> tuple[np.ndarray, dict]:
    """1 차원 기둥 줄(간격 spacing) — 자리 · 선호 방향(+) 이웃(없으면 −1)."""
    idx = np.arange(n)
    e_idx = np.stack([np.where(idx - 1 >= 0, idx - 1, -1)] * 2, 1)
    s_idx = np.stack([np.where(idx + 1 < n, idx + 1, -1)] * 2, 1)
    return idx * spacing_deg, {"a": (e_idx, s_idx)}


def gauss_1d(pos: np.ndarray, fwhm_deg: float) -> np.ndarray:
    """1 차원 자리끼리 가우시안 무게(행 합 1)."""
    d = pos[:, None] - pos[None, :]
    w = np.exp(-0.5 * (d / (fwhm_deg / 2.355)) ** 2)
    return w / w.sum(axis=1, keepdims=True)


def line_v3(
    p: dict,
    v3: dict,
    spacing_deg: float,
    dt_ms: float,
    v_deg_s: float,
    lam_deg: float,
    contrast: float,
    c50: float | None,
    n: int = 64,
    t_ms: float = 2000.0,
) -> dict[str, float]:
    """1 차원 줄에서 앞단 v3 를 흘림 — 뒤 절반 · 가운데 절반 기둥의 출력 평균 · 정류 x 의 정점.

    c50 이 None 이면 정규화 없이(c50 을 잴 때). 사인 줄무늬 밝기 = 0.5 + 0.5 · 대비 · sin.
    """
    pos, nb = line(n, spacing_deg)
    web = Weber(gauss_1d(pos, float(v3["weber"]["fwhm_deg"])), float(v3["weber"]["floor"]))
    dn = v3["divnorm"]
    norm = (
        None
        if c50 is None
        else DivNorm(
            gauss_1d(pos, float(dn["pool_fwhm_deg"])), float(dn["p"]), float(dn["w_pool"]), c50
        )
    )
    det = Hrc(n, nb, dt_ms, p, norm)
    mid = slice(n // 4, 3 * n // 4)
    steps = int(t_ms / dt_ms)
    outs, xs = [], []
    for k in range(steps):
        lum = 0.5 + 0.5 * contrast * np.sin(
            2 * np.pi * (pos - v_deg_s * k * dt_ms / 1000.0) / lam_deg
        )
        r = det.step(web(lum))["on"]["a"]
        if k >= steps // 2:
            outs.append(float(r[mid].mean()))
            xs.append(float(np.maximum(det.hp, 0.0)[mid].max()))  # 정규화 전 정류 x(ON)
    return {"out": float(np.mean(outs)), "x_peak": float(np.max(xs))}


class Hrc:
    """옆쌍 상관 검출기(기둥 × 4 방향 × ON/OFF) — 상태를 가진다.

    설정 frontend.t4t5 · model hrc-adjacent.

    입력 x = 정류(고역(L, τ_hp)) — ON 은 양의 쪽 · OFF 는 음의 쪽. 저역 l = LP(x, τ_lp).
    선호 방향(−1 → +1)의 출력 = ½[(l₋₁·x₀ − x₋₁·l₀) + (l₀·x₊₁ − x₀·l₊₁)] → 0 에서 자름.
    바로 옆 기둥끼리만 곱한다 — 앞 · 뒤 이웃을 바로 곱하면 17.6° 떨어져
    30° 줄무늬에서 방향이 뒤집힌다(10-17 시험).
    """

    def __init__(
        self, n_cols: int, nb: dict, dt_ms: float, p: dict, norm: DivNorm | None = None
    ) -> None:
        self.nb = nb
        self.norm = norm
        self.a_hp = float(np.exp(-dt_ms / float(p["tau_hp_ms"])))
        self.a_lp = float(np.exp(-dt_ms / float(p["tau_lp_ms"])))
        self.n_cols = n_cols
        self.reset()

    def reset(self) -> None:
        """상태(앞 그림 · 고역 · 저역)를 처음으로."""
        self.prev = None
        self.hp = np.zeros(self.n_cols)
        self.lp = {pol: np.zeros(self.n_cols) for pol in ("on", "off")}

    def step(self, lum: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        if self.prev is None:
            self.prev = lum.copy()
        self.hp = self.a_hp * (self.hp + lum - self.prev)  # 1 차 고역
        self.prev = lum.copy()
        out = {}
        for pol, sgn in (("on", 1.0), ("off", -1.0)):
            x = np.maximum(sgn * self.hp, 0.0)
            if self.norm is not None:
                x = self.norm(x)
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

    def __init__(self, g: Graph, face_px: int, dt_ms: float, cfg: dict | None = None) -> None:
        cfg = cfg if cfg is not None else config.load("eyeworld")["frontend"]
        self.cfg = cfg
        self.dt_ms = dt_ms
        self.r_max = float(cfg["r_max_hz"])
        faces = body_faces()
        self.cols = eye_columns(g)
        self.samplers, self.det, self.col_dirs = {}, {}, {}
        for s, c in self.cols.items():
            dirs = body_dir(0.0, c.az, c.el)
            self.col_dirs[s] = dirs
            self.samplers[s] = Sampler(faces, face_px, float(cfg["acceptance_sigma_deg"])).fit(dirs)
        self._detectors(self.spacing_deg())
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
        self.last_raw = np.zeros(len(self.t45))  # T4/T5 출력 ÷ ref(자르기 전 · 진단용)

    def reset(self) -> None:
        """상태(검출기 · 소엽 저역 · 앞 그림)를 처음으로 — 무게 · 기준은 그대로."""
        for d in self.det.values():
            d.reset()
        self.lob_prev = None
        self.lob_filt = np.zeros(len(self.lob))

    def _detectors(self, spacing: float) -> None:
        """기준 ref · 눈마다 상관 검출기(v2 = 밝기 그대로)."""
        p = self.cfg["t4t5"]
        self.ref = calibrate_ref(
            p, spacing, self.dt_ms, float(p["v_ref_deg_s"]), float(p["lambda_deg"])
        )
        for s, c in self.cols.items():
            self.det[s] = Hrc(len(c.h), neighbours(c), self.dt_ms, p)

    def _col_in(self, s: str, lum: np.ndarray) -> np.ndarray:
        """기둥 밝기 → 상관 검출기 입력(v2 = 밝기 그대로)."""
        return lum

    def _lob_in(self, key: tuple[str, str], lum: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """소엽 세포 수용장 밝기 → dL/dt 를 잴 신호(v2 = 밝기 그대로)."""
        return lum

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
        cols = {s: self.samplers[s].sample(eyes[s]) for s in ("L", "R")}
        r45 = np.zeros(len(self.t45))
        for s in ("L", "R"):
            out = self.det[s].step(self._col_in(s, cols[s]))
            for (pol, sub), ix in self.groups[s].items():
                r45[ix] = out[pol][sub][self.t45_col[ix]]
        self.last_raw = r45 / self.ref
        r45 = self.r_max * np.clip(self.last_raw, 0.0, 1.0)
        lum = np.zeros(len(self.lob))
        for key, (ix, smp) in self.lob_samplers.items():
            lum[ix] = self._lob_in(key, smp.sample(eyes[key[1]]), cols[key[1]])
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


class FrontendV3(FrontendV2):
    """앞단 v3 = v2 + (가) 국소 웨버 대비 + (나) 나눗셈 정규화(노트 18 §6.2 · 설정 frontend_v3).

    c50 = 대비 c50_contrast 사인 줄무늬(λ · v_ref)를 1 차원 기둥 줄에 흘렸을 때 정류 x 의 정점
    · ref = 대비 1 줄무늬(λ · v_ref)의 v3 출력 — 둘 다 앞단만으로 잼(뇌와 무관).
    소엽 세포 기준은 v2 와 같은 수(이제 웨버 대비 1 가장자리).
    """

    def __init__(self, g: Graph, face_px: int, dt_ms: float) -> None:
        ecfg = config.load("eyeworld")
        self.v3 = ecfg["frontend_v3"]
        super().__init__(g, face_px, dt_ms, cfg=ecfg[self.v3["base"]])
        fw, floor = float(self.v3["weber"]["fwhm_deg"]), float(self.v3["weber"]["floor"])
        self.lob_weber = {
            key: Weber(
                gauss_weights(
                    body_dir(0.0, self.lob.az[ix], self.lob.el[ix]), self.col_dirs[key[1]], fw
                ),
                floor,
            )
            for key, (ix, _) in self.lob_samplers.items()
        }

    def _detectors(self, spacing: float) -> None:
        p, v3, dn = self.cfg["t4t5"], self.v3, self.v3["divnorm"]
        v_ref, lam = float(p["v_ref_deg_s"]), float(p["lambda_deg"])
        args = (p, v3, spacing, self.dt_ms, v_ref, lam)
        self.c50 = line_v3(*args, float(dn["c50_contrast"]), None)["x_peak"]
        self.ref = line_v3(*args, 1.0, self.c50)["out"]
        fw, floor = float(v3["weber"]["fwhm_deg"]), float(v3["weber"]["floor"])
        self.weber = {}
        for s, c in self.cols.items():
            d = self.col_dirs[s]
            self.weber[s] = Weber(gauss_weights(d, d, fw), floor)
            norm = DivNorm(
                gauss_weights(d, d, float(dn["pool_fwhm_deg"])),
                float(dn["p"]),
                float(dn["w_pool"]),
                self.c50,
            )
            self.det[s] = Hrc(len(c.h), neighbours(c), self.dt_ms, p, norm)

    def _col_in(self, s: str, lum: np.ndarray) -> np.ndarray:
        return self.weber[s](lum)

    def _lob_in(self, key: tuple[str, str], lum: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.lob_weber[key](lum, around=cols)

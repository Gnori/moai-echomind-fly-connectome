"""T4/T5 뉴런의 시야 자리(eye-v1) — 기둥 좌표 → 방위 · 고도.

T4/T5 에는 공식 기둥 좌표가 없어 **좌표가 있는 입력 기둥의 시냅스 가중 평균**을 쓴다
(retino 모듈과 같은 식 · 10-16 확인 13,577/13,580).
방위 · 고도 변환은 retino-v2 규약 그대로: 두 눈 모든 기둥의 h · v 범위를
`azimuth_range_deg` · `elevation_range_deg` 로 선형
(오른눈 기준 · 왼눈은 방위 부호 반전 · + = 왼쪽) — 선언 · 근사.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config, retino
from ..graph import Graph
from ..loop.signals import side_of


@dataclass(frozen=True)
class Positions:
    """T4/T5 뉴런마다 자리. 배열은 모두 같은 길이."""

    idx: np.ndarray  # 그래프 인덱스
    kind: np.ndarray  # "T4" | "T5"
    sub: np.ndarray  # "a" | "b" | "c" | "d"
    side: np.ndarray  # "L" | "R"
    h: np.ndarray  # hex1 − hex2(앞 +)
    v: np.ndarray  # hex1 + hex2(위 +)
    az: np.ndarray  # 방위 °(+ 왼쪽 · 0 정면)
    el: np.ndarray  # 고도 °(+ 위)

    def __len__(self) -> int:
        return len(self.idx)

    def shuffled(self, seed: int) -> Positions:
        """아형 × 눈 안에서 자리(h · v · az · el)만 섞은 대응표 — 켜진 수 · 총 입력은 같다."""
        rng = np.random.default_rng(seed)
        perm = np.arange(len(self))
        for key in sorted(set(zip(self.kind, self.sub, self.side))):
            m = np.flatnonzero((self.kind == key[0]) & (self.sub == key[1]) & (self.side == key[2]))
            perm[m] = rng.permutation(m)
        return Positions(
            self.idx,
            self.kind,
            self.sub,
            self.side,
            self.h[perm],
            self.v[perm],
            self.az[perm],
            self.el[perm],
        )


def _linear(x: np.ndarray, lo_in: float, hi_in: float, lo_out: float, hi_out: float) -> np.ndarray:
    if hi_in == lo_in:
        return np.full_like(x, (lo_out + hi_out) / 2)
    return lo_out + (x - lo_in) / (hi_in - lo_in) * (hi_out - lo_out)


def type_positions(g: Graph, types: list[str]) -> Positions:
    """아무 기둥 종류나 — 공식 좌표가 있으면 그것, 없으면 입력 가중 평균(T4/T5 와 같은 식).

    `kind` 칸에 종류 이름 · `sub` 칸은 "-".
    """
    rcfg = config.load("retino")
    typ = g.columns["type"].astype(str)
    side = side_of(g)
    hexes = retino.column_hex(g)
    cols = hexes[~np.isnan(hexes[:, 0])]
    hs_all, vs_all = cols[:, 0] - cols[:, 1], cols.sum(1)
    idx = np.flatnonzero(np.isin(typ, types) & np.isin(side, ["L", "R"]))
    hx, w = retino.weighted_input_hex(g, idx, hexes)
    off = ~np.isnan(hexes[idx, 0])
    hx[off] = hexes[idx[off]]
    ok = (off | (w >= float(rcfg["min_input_synapses"]))) & ~np.isnan(hx[:, 0])
    idx, hx = idx[ok], hx[ok]
    h, v = hx[:, 0] - hx[:, 1], hx.sum(1)
    az = _linear(
        h, float(hs_all.min()), float(hs_all.max()), *map(float, rcfg["azimuth_range_deg"])
    )
    s = side[idx].astype(str)
    az = np.where(s == "L", -az, az)
    el = _linear(
        v, float(vs_all.min()), float(vs_all.max()), *map(float, rcfg["elevation_range_deg"])
    )
    return Positions(idx=idx, kind=typ[idx], sub=np.full(len(idx), "-"), side=s,
                     h=h, v=v, az=az, el=el)


def t4t5_positions(g: Graph, types: list[str] | None = None) -> Positions:
    """설정(eyeworld.yaml eye · retino.yaml)대로 T4/T5 자리를 계산한다.

    좌표를 못 얻은 뉴런은 뺀다.
    """
    eye = config.load("eyeworld")["eye"]
    rcfg = config.load("retino")
    types = types or list(eye["t4t5_types"])
    typ = g.columns["type"].astype(str)
    side = side_of(g)
    hexes = retino.column_hex(g)
    cols = hexes[~np.isnan(hexes[:, 0])]
    hs_all, vs_all = cols[:, 0] - cols[:, 1], cols.sum(1)
    idx = np.flatnonzero(np.isin(typ, types) & np.isin(side, ["L", "R"]))
    hx, w = retino.weighted_input_hex(g, idx, hexes)
    ok = (w >= float(rcfg["min_input_synapses"])) & ~np.isnan(hx[:, 0])
    idx, hx = idx[ok], hx[ok]
    h, v = hx[:, 0] - hx[:, 1], hx.sum(1)
    az = _linear(
        h, float(hs_all.min()), float(hs_all.max()), *map(float, rcfg["azimuth_range_deg"])
    )
    s = side[idx].astype(str)
    az = np.where(s == "L", -az, az)
    el = _linear(
        v, float(vs_all.min()), float(vs_all.max()), *map(float, rcfg["elevation_range_deg"])
    )
    t = typ[idx]
    return Positions(
        idx=idx,
        kind=np.array([x[:2] for x in t]),
        sub=np.array([x[2] for x in t]),
        side=s,
        h=h,
        v=v,
        az=az,
        el=el,
    )

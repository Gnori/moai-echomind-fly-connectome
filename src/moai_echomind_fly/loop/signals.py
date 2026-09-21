"""신호 벡터화 (R-3340 F-012 · I §3.2) — 초파리 회로 → 채널별 벡터.

구도: 초파리 회로 = 인코더(앞) · 투영기 = 다리 · Qwen = 생성기(뒤). 이 모듈은 인코더의 출력 쪽이다.
채널 = 주석 칸으로 고른 뉴런 묶음(config/channels.yaml). 단위 = 칸 조합 × 좌우.
전기 신호 = 창(300 ms)을 구간(3 × 100 ms)으로 나눠 단위마다 ① 발화율 ② 평균 탈분극.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import config
from ..graph import Graph
from ..sim import Engine


def side_of(graph: Graph) -> np.ndarray:
    """좌우: somaSide(L/R/M) → 없으면 rootSide(감각 뉴런은 세포체가 밖이라 신경 뿌리 쪽)
    → 없으면 instance 끝의 _L/_R → 그래도 없으면 M."""
    soma = graph.columns["somaSide"]
    root = graph.columns.get("rootSide", np.full(graph.n, "", dtype=object))
    inst = graph.columns["instance"]
    side = np.where(np.isin(soma, ["L", "R", "M"]), soma, "")
    side = np.where((side == "") & np.isin(root, ["L", "R", "M"]), root, side)
    tail = np.array([("L" if s.endswith("_L") else "R" if s.endswith("_R") else "M")
                     for s in inst], dtype=object)
    return np.where(side == "", tail, side)


@dataclass
class Channel:
    name: str
    neurons: np.ndarray            # 그래프 인덱스
    unit_of: np.ndarray            # 뉴런마다 단위 번호
    units: list[str]               # "LC4:L" …
    unit_size: np.ndarray          # 단위마다 뉴런 수

    @property
    def dim(self) -> int:
        return len(self.units)


@dataclass
class ChannelTable:
    version: str
    window_ms: float
    bins: int
    rate_scale_hz: float
    depol_scale_mV: float
    channels: dict[str, Channel]
    all_neurons: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @classmethod
    def build(cls, graph: Graph, cfg: dict[str, Any] | None = None,
              exclude: np.ndarray | None = None) -> ChannelTable:
        """`exclude` = 우리가 직접 자극하는 뉴런(감각 입구 · 하강 뉴런 명령). 벡터에서 뺀다 —
        그래야 벡터가 우리 입력을 되읽는 것이 아니라 **회로를 거쳐 생긴 활동**만 담는다."""
        cfg = cfg or config.load("channels")
        excluded = np.zeros(graph.n, bool)
        if exclude is not None:
            excluded[exclude] = True
        side = side_of(graph)
        chans: dict[str, Channel] = {}
        for name, spec in cfg["channels"].items():
            if "select" in spec:
                mask = np.ones(graph.n, bool)
                for col, vals in spec["select"].items():
                    mask &= np.isin(graph.columns[col], vals)
            else:
                mask = np.zeros(graph.n, bool)
                for col, vals in spec["select_any"].items():
                    mask |= np.isin(graph.columns[col], vals)
            if spec.get("exclude_superclass"):
                mask &= ~np.isin(graph.columns["superclass"], spec["exclude_superclass"])
            mask &= ~excluded
            idx = np.flatnonzero(mask)
            keys = np.array([":".join([*(str(graph.columns[c][i]) or "-" for c in spec["unit"]),
                                       str(side[i])]) for i in idx], dtype=object)
            units, unit_of = np.unique(keys, return_inverse=True)
            chans[name] = Channel(name, idx, unit_of, [str(u) for u in units],
                                  np.bincount(unit_of, minlength=len(units)))
        table = cls(cfg["version"], float(cfg["window_ms"]), int(cfg["bins"]),
                    float(cfg["rate_scale_hz"]), float(cfg["depol_scale_mV"]), chans)
        table.all_neurons = np.unique(np.concatenate([c.neurons for c in chans.values()]))
        return table

    def describe(self) -> dict[str, Any]:
        """표마다 "주석에서 온 수 / 채운 수" (R-3340 F-012)."""
        return {name: {"neurons": int(len(c.neurons)), "units": c.dim,
                       "dim": 2 * self.bins * c.dim, "from_annotation": int(len(c.neurons)),
                       "filled": 0} for name, c in self.channels.items()}


@dataclass
class ChannelVector:
    name: str
    units: list[str]
    rate: np.ndarray               # [bins, units] Hz
    depol: np.ndarray              # [bins, units] mV (v − v0)

    def features(self, rate_scale_hz: float, depol_scale_mV: float) -> np.ndarray:
        r = np.log1p(self.rate) / np.log1p(rate_scale_hz)
        d = self.depol / depol_scale_mV
        return np.concatenate([r.ravel(), d.ravel()]).astype(np.float32)


class SignalReader:
    """엔진을 창 하나만큼 돌리며 채널 벡터를 모은다."""

    def __init__(self, engine: Engine, table: ChannelTable) -> None:
        self.eng = engine
        self.table = table
        engine.accumulate_voltage(table.all_neurons)
        self._pos = {name: np.searchsorted(table.all_neurons, c.neurons)
                     for name, c in table.channels.items()}

    @property
    def bin_ms(self) -> float:
        return self.table.window_ms / self.table.bins

    def run_bin(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """구간 하나(bin_ms)를 돌린다.

        Returns:
            (채널별 발화율, 채널별 탈분극, 이 구간에 발화한 뉴런)
        """
        t = self.table
        s0 = self.eng.step
        self.eng.run(self.bin_ms)
        _, neuron = self.eng.spikes_since(s0)
        vmean = self.eng.take_voltage_mean()
        counts = np.bincount(neuron, minlength=self.eng.n)
        v0 = self.eng.p.v0_mV
        rates, depol = {}, {}
        for name, c in t.channels.items():
            per = np.bincount(c.unit_of, weights=counts[c.neurons], minlength=c.dim)
            rates[name] = per / c.unit_size / (self.bin_ms / 1000.0)
            vm = np.bincount(c.unit_of, weights=vmean[self._pos[name]], minlength=c.dim)
            depol[name] = vm / c.unit_size - v0
        return rates, depol, neuron

    def assemble(self, bins: list[tuple[dict, dict]]) -> dict[str, ChannelVector]:
        t = self.table
        return {n: ChannelVector(n, c.units, np.stack([b[0][n] for b in bins]),
                                 np.stack([b[1][n] for b in bins]))
                for n, c in t.channels.items()}

    def window(self, before_bin=None) -> dict[str, ChannelVector]:
        """창(window_ms)을 돌린다. `before_bin(b)` 로 구간마다 입력을 바꿀 수 있다."""
        bins = []
        for b in range(self.table.bins):
            if before_bin is not None:
                before_bin(b)
            rates, depol, _ = self.run_bin()
            bins.append((rates, depol))
        return self.assemble(bins)

    def features(self, vectors: dict[str, ChannelVector]) -> dict[str, np.ndarray]:
        t = self.table
        return {n: v.features(t.rate_scale_hz, t.depol_scale_mV) for n, v in vectors.items()}


def text_summary(vectors: dict[str, ChannelVector], *, top: int = 4) -> str:
    """비교군 `text` 용 글 요약(06 §3.2 서식) — 채널마다 발화율 상위 단위."""
    lines = []
    for name, v in vectors.items():
        rate = v.rate.mean(axis=0)
        depol = v.depol.mean(axis=0)
        order = np.argsort(-rate)[:top]
        parts = [f"{v.units[i]} {rate[i]:.1f}Hz" for i in order if rate[i] > 0]
        if not parts:
            j = int(np.argmax(depol))
            parts = [f"(발화 없음 · 최대 탈분극 {v.units[j]} {depol[j]:+.1f}mV)"]
        lines.append(f"{name:9}: " + " · ".join(parts))
    return "\n".join(lines)

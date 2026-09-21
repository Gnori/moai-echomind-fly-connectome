"""감각 인코더 (R-3340 F-011) — 세계(몸 관측) → 초파리 감각 뉴런 발화율(Hz).

시각 입구는 광수용체가 아니라 LC/LPLC 다(R-3340 1.3-1 · 광수용체 경로 발화 0 실측).
모든 상수는 config/task.yaml 에 출처·선언과 함께 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..graph import Graph
from ..protocols import LoomSpec
from .signals import side_of
from .world import Obs


@dataclass
class SensoryMap:
    """감각 대응표 — 뉴런 인덱스 묶음(주석에서만 고른다 · 채운 수 0)."""
    stim_idx: np.ndarray                      # 자극받는 모든 감각 뉴런(엔진 set_poisson 순서)
    vision: dict[str, np.ndarray]             # "L"/"R" → (위치들, 상대 세기)
    vision_amp: dict[str, np.ndarray]
    smell: dict[str, np.ndarray]              # "L"/"R" → 위치들
    touch: dict[str, np.ndarray]              # "LF"…"RH" → 위치들

    def describe(self) -> dict[str, Any]:
        return {"vision": {s: int(len(v)) for s, v in self.vision.items()},
                "smell": {s: int(len(v)) for s, v in self.smell.items()},
                "touch": {k: int(len(v)) for k, v in self.touch.items()},
                "from_annotation": int(len(self.stim_idx)), "filled": 0}


def build_map(graph: Graph, cfg: dict[str, Any], loom: LoomSpec) -> SensoryMap:
    enc = cfg["encoder"]
    side = side_of(graph)
    groups: list[np.ndarray] = []

    def take(idx: np.ndarray) -> np.ndarray:
        start = sum(len(g) for g in groups)
        groups.append(idx)
        return np.arange(start, start + len(idx))

    vision, vision_amp = {}, {}
    for s in ("L", "R"):
        idx, amp = [], []
        for typ, a in loom.amplitude.items():
            sel = graph.select(type=typ)
            sel = sel[side[sel] == s]
            idx.append(sel)
            amp.append(np.full(len(sel), a))
        vision[s] = take(np.concatenate(idx))
        vision_amp[s] = np.concatenate(amp)
    smell = {}
    for s in ("L", "R"):
        sel = graph.select(type=enc["smell"]["types"])
        smell[s] = take(sel[side[sel] == s])
    touch = {}
    nerves = enc["touch"]["nerves"]
    for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
        sel = graph.select(klass=enc["touch"]["klass"], entryNerve=nerves[leg[1]])
        touch[leg] = take(sel[side[sel] == leg[0]])
    return SensoryMap(np.concatenate(groups), vision, vision_amp, smell, touch)


class Encoder:
    def __init__(self, smap: SensoryMap, cfg: dict[str, Any], loom: LoomSpec) -> None:
        self.m = smap
        self.enc = cfg["encoder"]
        self.loom = loom

    def rates(self, obs: Obs) -> np.ndarray:
        """자극 뉴런(stim_idx 순서)의 발화율(Hz)."""
        r = np.zeros(len(self.m.stim_idx))
        v = self.enc["vision"]
        lo = obs.looming
        if lo["present"] and lo["expansion_deg_s"] > 0:
            gain = min(v["gain_max"], lo["expansion_deg_s"] / v["expansion_ref_deg_s"])
            left = 1.0 / (1.0 + math.exp(-lo["azimuth_deg"] / v["side_softness_deg"]))
            for s, w in (("L", left), ("R", 1.0 - left)):
                r[self.m.vision[s]] = self.m.vision_amp[s] * self.loom.peak_hz * gain * w
        sm = self.enc["smell"]
        for s in ("L", "R"):
            c = obs.odor_sensors[sm["sensors"][s]]
            top = math.log1p(sm["conc_max"] / sm["conc_ref"])
            level = math.log1p(max(c, 0.0) / sm["conc_ref"]) / top
            r[self.m.smell[s]] = sm["rate_max_hz"] * min(1.0, level)
        t = self.enc["touch"]
        for leg, pos in self.m.touch.items():
            f = obs.leg_contact.get(leg, 0.0)
            r[pos] = t["rate_max_hz"] * min(1.0, f / t["force_ref"])
        return r

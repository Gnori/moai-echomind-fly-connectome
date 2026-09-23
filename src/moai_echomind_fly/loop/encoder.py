"""감각 인코더 (R-3340 F-011) — 세계(몸 관측) → 초파리 감각 뉴런 발화율(Hz).

시각 입구는 광수용체가 아니라 LC/LPLC 다(R-3340 1.3-1 · 광수용체 경로 발화 0 실측).
모든 상수는 config/task.yaml 에 출처·선언과 함께 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    vision_elev: dict[str, np.ndarray]        # "L"/"R" → LC 마다 고도(°) · 좌표 없으면 NaN
    smell: dict[str, np.ndarray]              # "L"/"R" → 위치들
    touch: dict[str, np.ndarray]              # "LF"…"RH" → 위치들
    vision_az: dict[str, np.ndarray] = field(default_factory=dict)   # retino-v2 방위 · 없으면 NaN

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

    elev_of, az_of = np.full(graph.n, np.nan), np.full(graph.n, np.nan)
    mode = enc["vision"].get("mode", "lr_sigmoid")
    if mode in ("retinotopic", "retinotopic2"):
        from .. import retino

        for n in retino.load(graph.meta.get("rewired_from", graph.name))["neurons"]:
            elev_of[n["idx"]] = n["elevation_deg"]
            az_of[n["idx"]] = n.get("azimuth_deg", np.nan)
        if mode == "retinotopic2" and np.isnan(az_of).all():
            raise ValueError("retinotopic2 는 방위가 있는 대응표(retino-v2 이상)가 필요하다")
    vision, vision_amp, vision_elev, vision_az = {}, {}, {}, {}
    for s in ("L", "R"):
        idx, amp = [], []
        for typ, a in loom.amplitude.items():
            sel = graph.select(type=typ)
            sel = sel[side[sel] == s]
            idx.append(sel)
            amp.append(np.full(len(sel), a))
        cat = np.concatenate(idx)
        vision[s] = take(cat)
        vision_amp[s] = np.concatenate(amp)
        vision_elev[s] = elev_of[cat]
        vision_az[s] = az_of[cat]
    smell = {}
    for s in ("L", "R"):
        sel = graph.select(type=enc["smell"]["types"])
        smell[s] = take(sel[side[sel] == s])
    touch = {}
    nerves = enc["touch"]["nerves"]
    for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
        sel = graph.select(klass=enc["touch"]["klass"], entryNerve=nerves[leg[1]])
        touch[leg] = take(sel[side[sel] == leg[0]])
    return SensoryMap(np.concatenate(groups), vision, vision_amp, vision_elev, smell, touch,
                      vision_az)


class Encoder:
    def __init__(self, smap: SensoryMap, cfg: dict[str, Any], loom: LoomSpec) -> None:
        self.m = smap
        self.enc = cfg["encoder"]
        self.loom = loom

    def vision_tune(self, s: str, elevation_deg: float, azimuth_deg: float) -> np.ndarray:
        """눈 s 의 LC 마다 시야 자리 조율(0~1). 좌표 없는 LC 는 1.

        lr_sigmoid: 1 · retinotopic(task-v2): 고도 가우시안 · retinotopic2: 고도 × 방위 가우시안.
        """
        v = self.enc["vision"]
        mode = v.get("mode", "lr_sigmoid")
        n = len(self.m.vision[s])
        if mode == "lr_sigmoid":
            return np.ones(n)
        ne = self.m.vision_elev[s]
        sig_el = float(v["elevation_sigma_deg"])
        tune = np.where(np.isnan(ne), 1.0, np.exp(-0.5 * ((elevation_deg - ne) / sig_el) ** 2))
        if mode == "retinotopic2":
            na = self.m.vision_az.get(s, np.full(n, np.nan))
            tune = tune * np.where(np.isnan(na), 1.0,
                                   np.exp(-0.5 * ((azimuth_deg - na)
                                                  / float(v["azimuth_sigma_deg"])) ** 2))
        return tune

    def rates(self, obs: Obs) -> np.ndarray:
        """자극 뉴런(stim_idx 순서)의 발화율(Hz)."""
        r = np.zeros(len(self.m.stim_idx))
        v = self.enc["vision"]
        lo = obs.looming
        if lo["present"] and lo["expansion_deg_s"] > 0:
            gain = min(v["gain_max"], lo["expansion_deg_s"] / v["expansion_ref_deg_s"])
            left = 1.0 / (1.0 + math.exp(-lo["azimuth_deg"] / v["side_softness_deg"]))
            el = float(lo.get("elevation_deg", 0.0))
            for s, w in (("L", left), ("R", 1.0 - left)):
                # LC 마다 시야 자리 조율 — 좌표 없는 LC 는 조율 없이 1(I-7469 v1.3)
                tune = self.vision_tune(s, el, float(lo["azimuth_deg"]))
                r[self.m.vision[s]] = self.m.vision_amp[s] * self.loom.peak_hz * gain * w * tune
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

"""행동 디코더 (R-3340 F-014 · I §3.3 · §3.4).

행동 JSON → 하강 뉴런 목표 발화율(포아송 자극) → **회로가 낸** 하강 뉴런 발화율 → FlyGym 좌우 구동.
반사 우선: 거대섬유(DNp01)가 발화하면 언어 모델 명령 대신 다음 구간을 멈춘다
(FlyGym 은 도약을 못 한다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..graph import Graph
from .signals import side_of

ACTION_KEYS = {"walk", "turn", "turn_strength", "stop", "groom", "note"}
IDLE = {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": False, "groom": False}
STOP = {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True, "groom": False}


def parse_action(text: str) -> tuple[dict | None, str]:
    """스키마(I §3.3) 검사. 통과하면 (행동, "") · 아니면 (None, 까닭)."""
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        a = json.loads(s)
    except ValueError:
        return None, "JSON 아님"
    if not isinstance(a, dict):
        return None, "객체 아님"
    extra = set(a) - ACTION_KEYS
    if extra:
        return None, f"카탈로그 밖 키 {sorted(extra)}"
    for k in ("walk", "turn", "stop"):
        if k not in a:
            return None, f"필수 키 없음 {k}"
    if a["turn"] not in ("left", "right", "none"):
        return None, "turn 값"
    try:
        out = {"walk": min(1.0, max(0.0, float(a["walk"]))), "turn": a["turn"],
               "turn_strength": min(1.0, max(0.0, float(a.get("turn_strength", 1.0)))),
               "stop": bool(a["stop"]), "groom": bool(a.get("groom", False))}
    except (TypeError, ValueError):
        return None, "숫자 값"
    return out, ""


@dataclass
class Decoder:
    graph: Graph
    cfg: dict[str, Any]
    dn_idx: np.ndarray = field(init=False)
    walk_pos: np.ndarray = field(init=False)
    turn_pos: dict[str, np.ndarray] = field(init=False)
    escape_idx: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        dec = self.cfg["decoder"]
        side = side_of(self.graph)
        cat = dec["catalog"]
        walk = self.graph.select(type=cat["walk"]["types"])
        tl = self.graph.select(type=cat["turn_left"]["types"])
        tl = tl[side[tl] == cat["turn_left"]["side"]]
        tr = self.graph.select(type=cat["turn_right"]["types"])
        tr = tr[side[tr] == cat["turn_right"]["side"]]
        self.dn_idx = np.concatenate([walk, tl, tr])
        self.walk_pos = np.arange(len(walk))
        self.turn_pos = {"L": np.arange(len(walk), len(walk) + len(tl)),
                         "R": np.arange(len(walk) + len(tl), len(self.dn_idx))}
        self.escape_idx = self.graph.select(type=dec["drive"]["escape_type"])

    def target_rates(self, action: dict) -> np.ndarray:
        """행동 → 하강 뉴런(dn_idx 순서) 목표 발화율."""
        m = self.cfg["decoder"]["max_rate_hz"]
        r = np.zeros(len(self.dn_idx))
        if action["stop"]:
            return r
        r[self.walk_pos] = action["walk"] * m
        if action["turn"] in ("left", "right"):
            r[self.turn_pos["L" if action["turn"] == "left" else "R"]] = action["turn_strength"] * m
        return r

    def drive(self, bin_spikes: np.ndarray, bin_ms: float) -> tuple[tuple[float, float], bool]:
        """회로가 이 구간에 낸 하강 뉴런 발화 → FlyGym 좌우 구동 · 반사 여부."""
        d = self.cfg["decoder"]["drive"]
        if np.isin(bin_spikes, self.escape_idx).any():
            return (0.0, 0.0), True
        counts = np.bincount(bin_spikes, minlength=self.graph.n)
        hz = counts[self.dn_idx] / (bin_ms / 1000.0)
        walk = hz[self.walk_pos].mean() / d["walk_rate_ref_hz"] if len(self.walk_pos) else 0.0
        tl = hz[self.turn_pos["L"]].mean() if len(self.turn_pos["L"]) else 0.0
        tr = hz[self.turn_pos["R"]].mean() if len(self.turn_pos["R"]) else 0.0
        turn = (tl - tr) / d["walk_rate_ref_hz"]
        left = float(np.clip(walk - d["turn_gain"] * turn, -0.5, 1.5))
        right = float(np.clip(walk + d["turn_gain"] * turn, -0.5, 1.5))
        return (left, right), False

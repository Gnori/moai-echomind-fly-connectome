"""물체 인식 시험(R-5138 S0b) — 에코월드 주변 전부를 초파리 감각으로.

- `scene_objects`: /exp/near 의 개체(사람 · 동물 · 사물 · 막는 것)마다 NPC 기준 방위(+ 왼쪽) ·
  거리(과제 mm) · 시각 크기 · 커지는 속도(앞 구간 크기 기억 · NPC 가 걸어 다가가도 커짐) · 종류.
- `MultiEncoder`: 지금 인코더의 시각 식(좌우 시그모이드 · 확대 속도 게인 · 고도 · 방위 조율)을
  물체마다 적용해 **더한다**(상한 = 한 물체 최대 × config 상한). 냄새 · 촉각은 그대로.
- `truth_labels`: 결정마다 참값 다섯(판정 기준 · R-5138 S0b) — 월드 좌표에서.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..loop.encoder import Encoder


def _size_m(o: dict[str, Any]) -> float:
    s = o.get("size") or {}
    if s.get("radius_m"):
        return float(s["radius_m"])
    if s.get("half_w_m") or s.get("half_d_m"):
        return float(max(s.get("half_w_m") or 0.0, s.get("half_d_m") or 0.0))
    if s.get("w_m") or s.get("d_m"):
        return 0.5 * float(max(s.get("w_m") or 0.0, s.get("d_m") or 0.0))
    return float(o.get("radius_m") or 0.3)


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class SceneMemory:
    """물체마다 앞 구간 시각 크기 · 거리 — 커지는 속도 · 다가옴 계산."""

    def __init__(self) -> None:
        self.prev: dict[str, tuple[float, float]] = {}

    def reset(self) -> None:
        self.prev = {}


def scene_objects(me: dict[str, Any], near: list[dict[str, Any]], scale_m_per_mm: float,
                  dt_s: float, mem: SceneMemory, skip: set[str]) -> list[dict[str, Any]]:
    """NPC(me · 우리 좌표 x, y · 세계 기준 방향)와 이웃 목록(우리 좌표) → 감각용 물체 목록."""
    out, seen = [], {}
    for o in near:
        if o["ref"] in skip:
            continue
        dx, dy = o["x"] - me["x"], o["y"] - me["y"]
        d_m = max(math.hypot(dx, dy), 1e-3)
        r_m = _size_m(o)
        size = math.degrees(2.0 * math.atan2(r_m, d_m))
        az = wrap(math.degrees(math.atan2(dy, dx)) - me["heading_deg"])
        p = mem.prev.get(o["ref"])
        exp = 0.0 if p is None or dt_s <= 0 else (size - p[0]) / dt_s
        rad_v = 0.0 if p is None or dt_s <= 0 else (d_m - p[1]) / dt_s     # − 면 다가옴
        seen[o["ref"]] = (size, d_m)
        out.append({"ref": o["ref"], "kind": o.get("kind"), "cls": o.get("cls") or o.get("kind"),
                    "azimuth_deg": az, "distance_m": d_m,
                    "distance_mm": d_m / scale_m_per_mm, "size_deg": size,
                    "expansion_deg_s": exp, "radial_m_s": rad_v,
                    "elevation_deg": 0.0, "present": True})
    mem.prev = seen
    return out


class MultiEncoder(Encoder):
    """시각 = 물체마다 지금 인코더 식을 적용해 더함 · 냄새 · 촉각은 그대로."""

    def rates(self, obs) -> np.ndarray:
        objs = (obs.extra or {}).get("objects")
        if objs is None:
            return super().rates(obs)
        base_lo = {"present": False, "azimuth_deg": 0.0, "size_deg": 0.0,
                   "expansion_deg_s": 0.0, "elevation_deg": 0.0, "distance_mm": None}
        r = super().rates(_with_looming(obs, base_lo))           # 냄새 · 촉각만
        vis = np.concatenate([self.m.vision["L"], self.m.vision["R"]])
        acc = np.zeros(len(vis))
        for o in objs:
            if o["expansion_deg_s"] <= 0:
                continue
            ro = super().rates(_with_looming(obs, o))
            acc += ro[vis] - r[vis]
        v = self.enc["vision"]
        cap = (float(self.loom.peak_hz) * float(v["gain_max"])
               * float(v.get("multi_cap", 1.0)))
        amp = np.concatenate([self.m.vision_amp["L"], self.m.vision_amp["R"]])
        r[vis] = r[vis] + np.minimum(acc, amp * cap)
        return r


def _with_looming(obs, lo: dict[str, Any]):
    from dataclasses import replace

    return replace(obs, looming=lo)


def truth_labels(objs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, str]:
    """참값 다섯(R-5138 S0b · 결과 전 잠금)."""
    near_r = float(cfg["near_m"])
    close_r = float(cfg["close_m"])
    if not objs:
        return {"present": "없음", "side": "없음", "close": "없음", "approaching": "없음",
                "person": "없음"}
    n = min(objs, key=lambda o: o["distance_m"])
    if n["distance_m"] > near_r:
        return {"present": "없음", "side": "없음", "close": "없음", "approaching": "없음",
                "person": "없음"}
    return {"present": "있음",
            "side": "왼쪽" if n["azimuth_deg"] >= 0 else "오른쪽",
            "close": "가까움" if n["distance_m"] <= close_r else "멂",
            "approaching": "다가옴" if n["radial_m_s"] < -float(cfg["approach_m_s"]) else "아님",
            "person": "사람" if n["cls"] == "person" else "그 밖"}

"""촉각(R-1177 F-011 · 10-18) — 몸 부위 × 좌우 촉각 털 무리 고르기 · 닿음 → 무리.

고르기: 연결 지도 주석만(지어내지 않음) — 부위 = 촉각 털이 들어오는 신경(entryNerve) ·
좌우 = 신경 뿌리 쪽 · 머리 털 = 뇌 쪽 기계감각 털(BM_* · 입 털 BM_Taste 뺌) ·
더듬이 털 = 기계감각 grooming(AN).
닿음 → 무리: 몸 기준 방위(+ 왼쪽)로 앞 · 옆 · 뒤 구역을 고르고
그 구역의 부위 무리를 닿은 쪽(좌우)에서 켠다.
구역 · 부위 대응은 감각기 모형(선언 · config) — 몸 행동 규칙이 아니다(몸은 하강 뉴런 그대로).
값은 config/eyeworld.yaml touch_entry(고르기) · view_brain.touch(닿음).
"""

from __future__ import annotations

import numpy as np

from ..loop.signals import side_of


def groups(g, cfg: dict) -> dict[str, np.ndarray]:
    """부위 × 좌우(이름 '<부위>_<L|R>') → 뉴런 인덱스. cfg = touch_entry.

    reference 무리는 비교 기준(감각 아님).
    """
    cls = g.columns["class"].astype(str)
    sub = g.columns["subclass"].astype(str)
    nerve = g.columns["entryNerve"].astype(str)
    typ = g.columns["type"].astype(str)
    side = side_of(g).astype(str)
    tact = np.isin(cls, cfg["tactile_class"])
    out: dict[str, np.ndarray] = {}
    for part, nerves in cfg["parts"].items():
        for s in "LR":
            out[f"{part}_{s}"] = np.flatnonzero(tact & np.isin(nerve, nerves) & (side == s))
    hc = cfg["head"]
    head = (np.isin(cls, hc["class"]) & np.char.startswith(typ, hc["type_prefix"])
            & ~np.isin(typ, hc["exclude"]))
    ant = np.isin(cls, cfg["antenna"]["class"]) & np.isin(sub, cfg["antenna"]["subclass"])
    for s in "LR":
        out[f"머리털_{s}"] = np.flatnonzero(head & (side == s))
        out[f"더듬이털_{s}"] = np.flatnonzero(ant & (side == s))
    for name, types in cfg.get("reference", {}).items():
        for s in "LR":
            out[f"[기준]{name}_{s}"] = np.flatnonzero(np.isin(typ, types) & (side == s))
    return out


def contact_groups(contacts: list[dict], tcfg: dict) -> list[str]:
    """닿음 목록([{bearing_deg(+ 왼쪽), gap_m}]) → 켤 무리 이름(정렬 · 겹침 없음).

    tcfg = view_brain.touch.
    """
    on: set[str] = set()
    for c in contacts:
        if float(c.get("gap_m", 0.0)) > float(tcfg["margin_m"]):
            continue
        b = float(c["bearing_deg"])
        b = (b + 180.0) % 360.0 - 180.0
        a = abs(b)
        sides = "LR" if a <= float(tcfg["center_deg"]) or a >= 180.0 - float(tcfg["center_deg"]) \
            else ("L" if b > 0 else "R")
        for sec in tcfg["sectors"]:
            if a <= float(sec["max_deg"]):
                on.update(f"{p}_{s}" for p in sec["parts"] for s in sides)
                break
    return sorted(on)

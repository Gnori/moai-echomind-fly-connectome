"""어두운 원판 자극 · 소엽 기둥 앞단(V0b · 설계 문서 I 3.5 · 연구 노트 17).

자극: 밝은 배경에 어두운 원판 — 다가옴(반지름 커짐) · 물러남(작아짐) ·
어두워지기만(크기 고정 · 대비만 오름).
밝기 L(p, t) = 1 − c(t) · S((r(t) − d(p)) / σ)
— d 는 원판 중심까지 각거리 · S 는 로지스틱(가장자리 흐림 σ).
소엽 기둥 종류마다 (극성 on|off · 방식 transient|sustained · 수용장 σ_rf) 로 반응:
  transient: 구간 사이 밝기 변화 ΔL/Δt 의 극성 쪽만
  sustained: 배경(1)에서 벗어난 정도의 극성 쪽만.
발화율 = r_max · clip(반응 / ref, 0, 1). 값은 설정(eyeworld.yaml v0b) — "미정"이면 실행을 거부한다.
"""

from __future__ import annotations

import numpy as np

from .geometry import Positions


def angular_distance(pos: Positions, center_deg: tuple[float, float]) -> np.ndarray:
    a, e = np.radians(pos.az), np.radians(pos.el)
    ca, ce = np.radians(center_deg[0]), np.radians(center_deg[1])
    cosd = np.sin(e) * np.sin(ce) + np.cos(e) * np.cos(ce) * np.cos(a - ca)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def disk_course(
    kind: str, r0: float, r1: float, contrast: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """구간 가운데 시각마다 (반지름 °, 대비). kind = loom | recede | dim."""
    f = (np.arange(n_bins) + 0.5) / n_bins
    if kind == "loom":
        return r0 + (r1 - r0) * f, np.full(n_bins, contrast)
    if kind == "recede":
        return r1 - (r1 - r0) * f, np.full(n_bins, contrast)
    if kind == "dim":
        return np.full(n_bins, r1), contrast * f
    raise ValueError(kind)


def luminance(d: np.ndarray, radius: float, contrast: float, sigma: float) -> np.ndarray:
    return 1.0 - contrast / (1.0 + np.exp(-(radius - d) / max(sigma, 1e-6)))


def lobula_rates(
    d: np.ndarray,
    radii: np.ndarray,
    contrasts: np.ndarray,
    bin_s: float,
    polarity: str,
    mode: str,
    sigma_deg: float,
    ref: float,
    r_max_hz: float,
    tau_ms: float = 0.0,
) -> list[np.ndarray]:
    """한 종류 뉴런들(각거리 d)의 구간별 발화율.

    polarity = on | off | onoff(|변화|) · tau_ms > 0 이면 반응에 1 차 시간 저역(선언).
    """
    lum = [luminance(d, r, c, sigma_deg) for r, c in zip(radii, contrasts)]
    a = float(np.exp(-bin_s * 1000.0 / tau_ms)) if tau_ms > 0 else 0.0
    out, prev, filt = [], np.ones_like(d), np.zeros_like(d)
    for cur in lum:
        if mode == "transient":
            x = (cur - prev) / bin_s
        elif mode == "sustained":
            x = cur - 1.0
        else:
            raise ValueError(mode)
        if polarity == "on":
            resp = np.maximum(x, 0.0)
        elif polarity == "off":
            resp = np.maximum(-x, 0.0)
        elif polarity == "onoff":
            resp = np.abs(x)
        else:
            raise ValueError(polarity)
        filt = a * filt + (1.0 - a) * resp
        out.append(r_max_hz * np.clip(filt / ref, 0.0, 1.0))
        prev = cur
    return out


def edge_ref(contrast: float, v_ref_deg_s: float, sigma_deg: float) -> float:
    """v_ref 로 움직이는 가장자리가 수용장(σ)을 지날 때 |dL/dt| 정점.

    이때 r_max — 결과와 무관하게 정하는 기준.
    """
    return contrast * v_ref_deg_s / (4.0 * sigma_deg)

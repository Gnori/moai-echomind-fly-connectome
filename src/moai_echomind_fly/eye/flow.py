"""분석식 광류 → T4/T5 발화율(V0 · 설계 문서 I 3.4 · 연구 노트 17 §8.1).

좌표: 몸 기준 단위 벡터 p = (cos el cos az, cos el sin az, sin el)
— x 앞 · y 왼 · z 위 · 방위 + = 왼쪽.
국소 움직임은 시야 각속도(°/s)로 (방위 성분, 고도 성분)에 둔다. 아형 선호 방향은 눈 격자 축
(a = −h 뒤로 · b = +h 앞으로 · c = +v 위 · d = −v 아래)이고,
h 는 오른눈에서 방위와 같은 쪽 · 왼눈에서 반대쪽이다.
발화율 = r_max · clip(국소 움직임 · 선호 방향 / v_ref, 0, 1) — T4 와 T5 에 같은 값(V0).
"""

from __future__ import annotations

import numpy as np

from .geometry import Positions

AXIS = {"a": (-1.0, 0.0), "b": (1.0, 0.0), "c": (0.0, 1.0), "d": (0.0, -1.0)}


def _unit(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def _basis(az_deg: np.ndarray, el_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p 에서 방위 증가 · 고도 증가 쪽 단위 접선."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    e_az = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)
    e_el = np.stack([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)], axis=-1)
    return e_az, e_el


def rotation_field(
    pos: Positions, speed_deg_s: float, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """세계가 몸 둘레로 위에서 보아 시계(cw) · 반시계(ccw)로 돎 — 국소 (방위, 고도) 각속도 °/s."""
    sign = -1.0 if direction == "cw" else 1.0  # 시계 = z 축 음의 회전(앞 점이 오른쪽으로)
    omega = np.array([0.0, 0.0, sign * speed_deg_s])
    p = _unit(pos.az, pos.el)
    vel = np.cross(omega, p)
    e_az, e_el = _basis(pos.az, pos.el)
    return (vel * e_az).sum(-1), (vel * e_el).sum(-1)


def loom_field(
    pos: Positions,
    center_deg: tuple[float, float],
    radius_deg: float,
    drdt_deg_s: float,
    sigma_deg: float,
    tangential: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """원판 가장자리 띠의 바깥 방사 움직임(tangential=True 면 같은 자리에서 접선 · 돌기)."""
    p = _unit(pos.az, pos.el)
    c = _unit(np.array(center_deg[0]), np.array(center_deg[1]))
    cosd = np.clip(p @ c, -1.0, 1.0)
    d = np.degrees(np.arccos(cosd))
    t = p * cosd[:, None] - c  # c 에서 멀어지는 쪽(접평면)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.divide(t, n, out=np.zeros_like(t), where=n > 1e-9)
    if tangential:
        t = np.cross(p, t)
    w = np.exp(-((d - radius_deg) ** 2) / (2.0 * sigma_deg**2))
    e_az, e_el = _basis(pos.az, pos.el)
    speed = (w * drdt_deg_s)[:, None]
    return (speed * t * e_az).sum(-1), (speed * t * e_el).sum(-1)


def rates(
    pos: Positions, m_az: np.ndarray, m_el: np.ndarray, r_max_hz: float, v_ref_deg_s: float
) -> np.ndarray:
    """국소 움직임 → 뉴런마다 발화율 Hz."""
    m_h = np.where(pos.side == "R", m_az, -m_az)
    ax = np.array([AXIS[s] for s in pos.sub])
    proj = ax[:, 0] * m_h + ax[:, 1] * m_el
    return r_max_hz * np.clip(proj / v_ref_deg_s, 0.0, 1.0)


def match_totals(
    pos: Positions, target: np.ndarray, source: np.ndarray, r_max_hz: float
) -> np.ndarray:
    """source 의 (종류 · 아형 · 눈) 묶음별 총 발화율을 target 과 같게 배율(상한 r_max 로 자름)."""
    out = source.copy()
    for key in set(zip(pos.kind, pos.sub, pos.side)):
        m = (pos.kind == key[0]) & (pos.sub == key[1]) & (pos.side == key[2])
        s, t = source[m].sum(), target[m].sum()
        if s > 0:
            out[m] = np.minimum(source[m] * (t / s), r_max_hz)
    return out

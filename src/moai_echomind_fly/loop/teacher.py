"""교사 정책과 캡션 (R-3340 F-017) — 세계의 **참값**에서 코드가 만든 정답.

① 정렬 단계의 정답 = 캡션(상황 설명 글) — "Qwen 이 초파리 뇌 활동을 읽어 내는가"를 잰다.
② 행동 단계의 정답 = 교사 행동 JSON — 먹이 냄새 쪽으로 · 다가오는 물체는 반대쪽으로.

⚠️ 이 규칙이 곧 결과를 정한다(R-3340 1.7) — 그래서 **섞은 배선 회로의 벡터로 똑같이 학습**한
대조군(`rewired`)과 나란히 잰다. 규칙은 전부 여기에 선언한다(정직 목록).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 선언된 문턱(정직 목록에 그대로 적힌다)
THREAT_DEG = 20.0          # 다가오는 물체의 각 크기가 이보다 크면 위협
NEAR_DEG, MID_DEG = 30.0, 10.0
ODOR_STRONG, ODOR_WEAK = 0.5, 0.05   # 두 더듬이 평균 농도(과제 단위)
SIDE_DEG = 15.0            # 방위가 ±15° 안이면 "앞"
REACH_MM = 2.0             # 먹이까지 이 거리 안이면 도착


@dataclass(frozen=True)
class Truth:
    """결정 창 끝의 세계 참값(몸 좌표 기준 방위: +는 왼쪽)."""
    looming_present: bool
    looming_azimuth_deg: float
    looming_size_deg: float
    looming_approaching: bool
    odor_left: float
    odor_right: float
    odor_azimuth_deg: float
    food_distance_mm: float
    legs_in_contact: int
    sound: bool = False
    female_present: bool = False
    female_azimuth_deg: float = 0.0
    # task-v2(3차원 · R-7469 F-008) — 자료에만 남긴다(캡션 서식은 그대로 · 3차원 캡션은 F-008 5단계)
    looming_elevation_deg: float = 0.0
    heading_deg: float = 0.0
    pos_xy: tuple[float, float] = (0.0, 0.0)
    body_euler_deg: tuple[float, ...] = ()   # FlyGym 순서 ZYX(방향 · 앞뒤 기울기 · 좌우 기울기)


def _side(az: float) -> str:
    return "앞" if abs(az) <= SIDE_DEG else ("왼쪽" if az > 0 else "오른쪽")


def caption(t: Truth) -> str:
    if t.looming_present:
        dist = ("가까움" if t.looming_size_deg >= NEAR_DEG
                else "중간" if t.looming_size_deg >= MID_DEG else "멀리")
        move = "다가옴" if t.looming_approaching else "머묾"
        loom = f"{_side(t.looming_azimuth_deg)} · {dist} · {move}"
    else:
        loom = "없음"
    mean = (t.odor_left + t.odor_right) / 2
    if mean < ODOR_WEAK:
        odor = "없음"
    else:
        level = "강함" if mean >= ODOR_STRONG else "약함"
        odor = f"{_side(t.odor_azimuth_deg)} · {level}"
    female = f"{_side(t.female_azimuth_deg)}" if t.female_present else "없음"
    return (f"다가오는 물체: {loom} / 먹이 냄새: {odor} / 다리 접촉: {t.legs_in_contact}개 / "
            f"소리: {'있음' if t.sound else '없음'} / 암컷 대용 물체: {female}")


def action(t: Truth) -> dict:
    """교사 행동(I §3.3 스키마). 위협이 먼저, 그다음 먹이."""
    if t.looming_present and t.looming_approaching and t.looming_size_deg >= THREAT_DEG:
        turn = "right" if t.looming_azimuth_deg >= 0 else "left"   # 물체 반대쪽
        return {"walk": 1.0, "turn": turn, "turn_strength": 1.0, "stop": False, "groom": False}
    if t.food_distance_mm <= REACH_MM:
        return {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True, "groom": False}
    if (t.odor_left + t.odor_right) / 2 >= ODOR_WEAK:
        az = t.odor_azimuth_deg
        if abs(az) <= SIDE_DEG:
            return {"walk": 0.8, "turn": "none", "turn_strength": 0.0, "stop": False,
                    "groom": False}
        strength = round(min(1.0, abs(az) / 90.0), 2)
        return {"walk": 0.6, "turn": "left" if az > 0 else "right", "turn_strength": strength,
                "stop": False, "groom": False}
    return {"walk": 0.5, "turn": "none", "turn_strength": 0.0, "stop": False, "groom": False}


def bearing_deg(dx: float, dy: float, heading_rad: float) -> float:
    """몸 좌표 기준 방위(+ 왼쪽). dx, dy = 목표 − 파리(세계 좌표)."""
    ang = math.atan2(dy, dx) - heading_rad
    return math.degrees(math.atan2(math.sin(ang), math.cos(ang)))


RULES = {
    "threat": f"다가오는 물체가 다가오고 각 크기 ≥ {THREAT_DEG}° 면 반대쪽으로 돌며 달린다",
    "reach": f"먹이까지 ≤ {REACH_MM} mm 면 멈춘다",
    "odor": f"냄새(평균 ≥ {ODOR_WEAK})가 있으면 그쪽으로 — ±{SIDE_DEG}° 안이면 곧장",
    "caption": f"거리 단계 가까움 ≥ {NEAR_DEG}° · 중간 ≥ {MID_DEG}° · 냄새 강함 ≥ {ODOR_STRONG}",
}

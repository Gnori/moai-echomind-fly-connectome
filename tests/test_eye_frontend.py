"""시각 앞단 v2(R-1177 · P eye Task 2) — 계산한 큐브맵으로 방향 선택성(그래프 필요)."""

import numpy as np
import pytest

from moai_echomind_fly.eye.compound import face_pixel_dirs
from moai_echomind_fly.eye.frontend import FrontendV2, body_faces

pytestmark = pytest.mark.data
N, DT = 16, 10.0


def _grating_frames(direction: str, steps: int) -> list[np.ndarray]:
    """몸 둘레 세로 줄무늬(λ 30°)가 위에서 보아 시계(cw: 방위 줄어듦) · 반시계로 90 °/s."""
    d = face_pixel_dirs(body_faces(), N)                       # 몸 축 = yaw 0 세계
    az = np.degrees(np.arctan2(d[..., 0], d[..., 2]))          # + = 왼쪽(x)
    sgn = 1.0 if direction == "cw" else -1.0
    out = []
    for k in range(steps):
        lum = 0.5 + 0.5 * np.sin(2 * np.pi * (az + sgn * 90.0 * k * DT / 1000.0) / 30.0)
        face = np.round(255 * lum).astype(np.uint8)
        out.append(np.stack([face, face]))                     # 두 눈 같은 그림(몸 축)
    return out


@pytest.fixture(scope="module")
def fe():
    from moai_echomind_fly import graph as graph_mod
    return FrontendV2(graph_mod.load(), N, DT)


def _mean_by(fe, rates):
    t = fe.t45
    r = rates[:len(t)]
    return {(s, k + sub): float(r[(t.side == s) & (t.kind == k) & (t.sub == sub)].mean())
            for s in "LR" for k in ("T4", "T5") for sub in "abcd"}


def test_rotation_drives_back_on_right_and_front_on_left(fe):
    for frame in _grating_frames("cw", 60):
        r = fe.step(frame)
    m = _mean_by(fe, r)
    assert m[("R", "T4a")] > 5 * m[("R", "T4b")]              # 시계: 오른눈 앞 → 뒤
    assert m[("L", "T4b")] > 5 * m[("L", "T4a")]              # 왼눈 뒤 → 앞
    assert m[("R", "T4c")] < 0.2 * m[("R", "T4a")]           # 위아래는 작게
    assert r.min() >= 0 and r.max() <= 150.0

"""앞단 v3 부품(R-1177 · 노트 18 §6.2) — 웨버 대비 · 나눗셈 정규화 · 1 차원 줄(그래프 없이)."""

import numpy as np

from moai_echomind_fly import config
from moai_echomind_fly.eye.frontend import DivNorm, Weber, gauss_1d, line, line_v3

CFG = config.load("eyeworld")
P, V3 = CFG["frontend"]["t4t5"], CFG["frontend_v3"]
SPACING, DT = 5.0, 10.0


def test_weber_ignores_overall_brightness():
    pos, _ = line(40, SPACING)
    w = Weber(gauss_1d(pos, 25.0), 0.02)
    lum = 0.4 + 0.2 * np.sin(pos / 7.0)
    assert np.allclose(w(lum), w(3.0 * lum))  # 밝기를 몇 배로 해도 대비는 같다


def test_divnorm_saturates_and_is_zero_at_rest():
    pos, _ = line(40, SPACING)
    dn = DivNorm(gauss_1d(pos, 30.0), 1.2, 2.0, 0.05)
    assert np.all(dn(np.zeros(40)) == 0.0)
    x = np.full(40, 1.0)
    assert np.allclose(dn(x), 1.0 / (0.05**1.2 + 1.0 + 2.0**1.2), rtol=1e-9)
    assert np.allclose(dn(10 * x), dn(x), rtol=0.02)  # 높은 대비에서는 대비에 거의 무관
    assert dn(np.full(40, 0.001))[0] < 0.1 * dn(x)[0]  # c50 보다 훨씬 작으면 작게


def test_line_v3_compresses_contrast_and_keeps_direction():
    args = (P, V3, SPACING, DT, 100.0, 30.0)
    c50 = line_v3(*args, float(V3["divnorm"]["c50_contrast"]), None)["x_peak"]
    assert c50 > 0
    top = line_v3(*args, 1.0, c50)["out"]
    low = line_v3(*args, 0.2, c50)["out"]
    assert low / top > 0.2  # 상관만이면 대비 제곱 0.04
    null = line_v3(P, V3, SPACING, DT, -100.0, 30.0, 1.0, c50)["out"]
    assert null < 0.2 * top

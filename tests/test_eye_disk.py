"""어두운 원판 · 소엽 앞단(R-1177 V0b) — 자극 꼴 · 극성 · 방식."""

import numpy as np

from moai_echomind_fly.eye import disk


def test_disk_courses():
    r, c = disk.disk_course("loom", 5, 30, 1.0, 3)
    assert r[0] < r[1] < r[2] and np.allclose(c, 1.0)
    r, c = disk.disk_course("recede", 5, 30, 1.0, 3)
    assert r[0] > r[2]
    r, c = disk.disk_course("dim", 5, 30, 1.0, 3)
    assert np.allclose(r, 30) and c[0] < c[2]


def test_luminance_dark_inside_bright_outside():
    lum = disk.luminance(np.array([0.0, 60.0]), 20.0, 1.0, 2.5)
    assert lum[0] < 0.01 and lum[1] > 0.99


def test_off_transient_fires_where_edge_passes_and_on_does_not():
    d = np.array([3.0, 15.0, 60.0])  # 처음부터 안 · 가장자리가 지나감 · 밖
    radii, contrasts = disk.disk_course("loom", 5, 30, 1.0, 30)
    off = np.array(
        disk.lobula_rates(d, radii, contrasts, 0.01, "off", "transient", 2.5, 50.0, 150.0)
    )
    on = np.array(disk.lobula_rates(d, radii, contrasts, 0.01, "on", "transient", 2.5, 50.0, 150.0))
    assert off[:, 1].max() > 0 and off[1:, 2].max() < 1e-3 and on.max() == 0.0
    sus = np.array(
        disk.lobula_rates(d, radii, contrasts, 0.01, "off", "sustained", 2.5, 0.5, 150.0)
    )
    assert sus[-1, 0] > 0 and sus[-1, 1] > 0 and sus[-1, 2] < 1e-2


def test_onoff_lowpass_and_edge_ref():
    d = np.array([15.0])
    radii, contrasts = disk.disk_course("loom", 5, 30, 1.0, 30)
    raw = np.array(
        disk.lobula_rates(d, radii, contrasts, 0.01, "onoff", "transient", 3.5, 7.0, 150.0)
    )
    lp = np.array(
        disk.lobula_rates(
            d, radii, contrasts, 0.01, "onoff", "transient", 3.5, 7.0, 150.0, tau_ms=40.0
        )
    )
    assert raw.max() > 0 and lp.max() > 0
    assert lp.argmax() >= raw.argmax()  # 저역은 늦게 정점
    assert np.isclose(disk.edge_ref(1.0, 100.0, 3.48), 100.0 / (4 * 3.48))

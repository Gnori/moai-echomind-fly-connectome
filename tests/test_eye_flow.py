"""분석식 광류 → T4/T5 발화율(R-1177 V0) — 방향 · 눈 · 아형 규약과 대조 장치."""

import numpy as np

from moai_echomind_fly.eye import flow
from moai_echomind_fly.eye.geometry import Positions


def _pos(rows):
    kind, sub, side, az, el = zip(*rows)
    n = len(rows)
    return Positions(
        idx=np.arange(n),
        kind=np.array(kind),
        sub=np.array(sub),
        side=np.array(side),
        h=np.zeros(n),
        v=np.zeros(n),
        az=np.array(az, float),
        el=np.array(el, float),
    )


def test_rotation_cw_is_back_on_right_eye_and_front_on_left_eye():
    p = _pos(
        [
            ("T4", "a", "R", -90, 0),
            ("T4", "b", "R", -90, 0),
            ("T4", "a", "L", 90, 0),
            ("T4", "b", "L", 90, 0),
        ]
    )
    m_az, m_el = flow.rotation_field(p, 90.0, "cw")
    r = flow.rates(p, m_az, m_el, 150.0, 100.0)
    assert r[0] > 0 and r[1] == 0  # 오른눈: 앞 → 뒤(a)
    assert r[2] == 0 and r[3] > 0  # 왼눈: 뒤 → 앞(b)
    m_az2, m_el2 = flow.rotation_field(p, 90.0, "ccw")
    r2 = flow.rates(p, m_az2, m_el2, 150.0, 100.0)
    assert r2[0] == 0 and r2[1] > 0 and r2[2] > 0 and r2[3] == 0
    assert np.allclose(m_el, 0.0)


def test_loom_edge_moves_outward_and_tangential_keeps_place():
    # 왼쪽 앞 30° 중심 · 반지름 10° — 뒤쪽(방위 40°)은 뒤로(왼눈 a) · 위(고도 10°)는 위로(c)
    p = _pos(
        [
            ("T5", "a", "L", 40, 0),
            ("T5", "b", "L", 40, 0),
            ("T5", "c", "L", 30, 10),
            ("T5", "d", "L", 30, 10),
            ("T5", "a", "L", 90, 0),
        ]
    )
    m_az, m_el = flow.loom_field(p, (30.0, 0.0), 10.0, 80.0, 2.5)
    r = flow.rates(p, m_az, m_el, 150.0, 100.0)
    assert r[0] > 0 and r[1] == 0 and r[2] > 0 and r[3] == 0
    assert r[4] < 1e-6  # 가장자리 띠 밖은 거의 0
    t_az, t_el = flow.loom_field(p, (30.0, 0.0), 10.0, 80.0, 2.5, tangential=True)
    radial = np.hypot(m_az, m_el)
    assert np.allclose(np.hypot(t_az, t_el), radial, atol=1e-9)  # 같은 자리 · 같은 세기
    assert abs(t_az[0] * m_az[0] + t_el[0] * m_el[0]) < 1e-6  # 방향만 직각


def test_match_totals_equalises_group_sums():
    p = _pos([("T4", "a", "L", 0, 0), ("T4", "a", "L", 10, 0), ("T4", "c", "L", 0, 0)])
    target = np.array([10.0, 20.0, 5.0])
    source = np.array([1.0, 1.0, 0.0])
    out = flow.match_totals(p, target, source, 150.0)
    assert np.isclose(out[:2].sum(), 30.0) and out[2] == 0.0


def test_shuffled_map_keeps_counts_per_subtype_and_eye():
    p = _pos([("T4", "a", "L", a, 0) for a in (0, 10, 20)] + [("T4", "a", "R", -5, 0)])
    s = p.shuffled(3)
    assert sorted(s.az[:3]) == [0, 10, 20] and s.az[3] == -5
    assert (s.idx == p.idx).all() and (s.sub == p.sub).all()

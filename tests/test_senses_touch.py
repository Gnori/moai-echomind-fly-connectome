"""촉각 무리 고르기 · 닿음 → 무리(R-1177 F-011 · 10-18)."""

from types import SimpleNamespace

import numpy as np

from moai_echomind_fly import config
from moai_echomind_fly.senses.touch import contact_groups, groups

TCFG = config.load("eyeworld")["view_brain"]["touch"]


def test_front_left_touch_turns_on_left_head_and_front_leg():
    assert contact_groups([{"bearing_deg": 30.0, "gap_m": 0.0}], TCFG) == ["머리털_L", "앞다리_L"]


def test_dead_ahead_touch_turns_on_both_sides():
    on = contact_groups([{"bearing_deg": 3.0, "gap_m": 0.0}], TCFG)
    assert on == ["머리털_L", "머리털_R", "앞다리_L", "앞다리_R"]


def test_side_and_rear_sectors_and_right_side():
    def at(b):
        return contact_groups([{"bearing_deg": b, "gap_m": 0.0}], TCFG)

    assert at(-90.0) == ["가운데다리_R", "날개·앞등_R"]
    assert at(150.0) == ["뒷다리_L", "뒷등_L"]
    assert at(-200.0) == ["뒷다리_L", "뒷등_L"]   # −200° = 왼쪽 뒤 160°
    assert at(175.0) == ["뒷다리_L", "뒷다리_R", "뒷등_L", "뒷등_R"]   # 정뒤 근처 = 좌우 둘 다


def test_gap_beyond_margin_is_no_touch():
    gap = float(TCFG["margin_m"]) + 0.01
    assert contact_groups([{"bearing_deg": 0.0, "gap_m": gap}], TCFG) == []
    assert contact_groups([], TCFG) == []


def test_groups_select_by_nerve_side_and_head_type():
    cfg = config.load("eyeworld")["touch_entry"]
    rows = [  # class · subclass · entryNerve · rootSide · type
        ("mechanosensory_tactile", "mechanosensory bristle", "ProLN", "L", "SNta"),
        ("mechanosensory_tactile", "mechanosensory bristle", "ProLN", "R", "SNta"),
        ("mechanosensory_tactile", "mechanosensory bristle", "MetaLN", "L", "SNta"),
        ("mechanosensory", "", "MxLbN", "L", "BM_InOm"),
        ("mechanosensory", "", "MxLbN", "L", "BM_Taste"),       # 입 털은 뺌
        ("mechanosensory", "grooming", "AN", "R", "JO-x"),
        ("visual", "", "", "L", "T4a"),
    ]
    cols = {k: np.array([r[i] for r in rows], dtype=object)
            for i, k in enumerate(["class", "subclass", "entryNerve", "rootSide", "type"])}
    cols["somaSide"] = np.array([""] * len(rows), dtype=object)
    cols["instance"] = np.array([""] * len(rows), dtype=object)
    g = SimpleNamespace(columns=cols, n=len(rows))
    out = groups(g, cfg)
    assert out["앞다리_L"].tolist() == [0] and out["앞다리_R"].tolist() == [1]
    assert out["뒷다리_L"].tolist() == [2]
    assert out["머리털_L"].tolist() == [3]
    assert out["더듬이털_R"].tolist() == [5]
    assert out["[기준]눈T4a·T5a_L"].tolist() == [6]

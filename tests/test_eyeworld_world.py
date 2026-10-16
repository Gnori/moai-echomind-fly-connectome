"""세계 서버(R-1177 · 설계 문서 I 2.7) — 좌표 규약 · 재현 · 충돌 · 대본 · 눈 · 장면 허가."""

import numpy as np
import pytest

from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.safety import SafetyError
from moai_echomind_fly.eyeworld.world import Body, Command, Entity, World, body_dir, forward, left


def _agent(**kw):
    return Entity(id="fly", kind="agent", asset=None, pos=np.zeros(3), radius=0.35,
                  body=Body(), **kw)


def test_axes_and_body_directions():
    assert np.allclose(forward(0.0), [0, 0, 1]) and np.allclose(left(0.0), [1, 0, 0])
    assert np.allclose(forward(np.pi / 2), left(0.0))                       # yaw + = 왼쪽으로 돎
    d = body_dir(0.0, np.array([0.0, 90.0, 0.0]), np.array([0.0, 0.0, 90.0]))
    assert np.allclose(d, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)    # 정면 · 왼 · 위


def test_same_seed_same_commands_same_trajectory():
    def run():
        w = World([_agent(), Entity(id="r", kind="prop", asset="x", pos=np.array([0.0, 0, 3]),
                                    radius=0.5)], seed=1)
        for k in range(200):
            w.step(0.01, {"fly": Command(speed=1.0, turn=30.0 * np.sin(k / 20))})
        return w.state_hash()
    assert run() == run()


def test_limits_and_collision_push_out():
    w = World([_agent(), Entity(id="wall", kind="prop", asset="x", pos=np.array([0.0, 0, 1.0]),
                                radius=0.5)])
    for _ in range(300):
        w.step(0.01, {"fly": Command(speed=10.0)})                          # 상한 4.5 로 잘림
    gap = np.linalg.norm(w.ents["wall"].pos - w.ents["fly"].pos)
    assert gap >= 0.35 + 0.5 - 1e-6 and w.ents["wall"].pos[2] == 1.0       # 몸만 밀려남
    w.step(0.01, {"fly": Command(turn=1000.0)})
    assert np.isclose(w.ents["fly"].yaw, np.radians(180.0) * 0.01)          # 회전 상한


def test_approach_script_heads_to_start_point_and_records_contact():
    haz = Entity(id="h", kind="person", asset="p", pos=np.array([0.0, 0, 5.0]), radius=0.3,
                 solid=False, script={"kind": "approach", "target": "fly", "speed": 2.5})
    w = World([_agent(), haz])
    hit = []
    for _ in range(300):
        hit += w.step(0.01)
    assert ("fly", "h") in hit and np.isclose(w.ents["h"].yaw, np.pi)


def test_eye_rig_faces_and_sides():
    w = World([_agent(yaw=np.pi / 2)])
    rig = w.eye_rig("fly")
    assert [r["side"] for r in rig] == ["L", "R"]
    front = rig[0]["faces"][0]
    assert front["name"] == "front" and np.allclose(front["forward"], [1, 0, 0])
    assert rig[0]["pos"][2] < rig[1]["pos"][2]                              # yaw 90°: 왼 = −z 쪽
    assert np.isclose(rig[0]["pos"][1], 1.6 * 0.936)


def test_scene_license_checks(tmp_path):
    (tmp_path / "a.glb").write_bytes(b"x")
    (tmp_path / "a.json").write_text('{"source": {"license": "CC0"}}', encoding="utf-8")
    ok = {"assets": {"a": {"path": "a.glb", "record": "a.json", "license": "CC0"}}}
    assert scene_mod.verify_assets(ok, tmp_path)["a"]["record_license"] == "CC0"
    bad = {"assets": {"a": {"path": "a.glb", "record": "a.json", "license": "CC-BY"}}}
    with pytest.raises(SafetyError):
        scene_mod.verify_assets(bad, tmp_path)
    nosrc = {"assets": {"a": {"path": "a.glb", "license": "CC0"}}}
    with pytest.raises(SafetyError):
        scene_mod.verify_assets(nosrc, tmp_path)

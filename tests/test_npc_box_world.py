"""R-5138 — 상자 월드 클라이언트(가짜 HTTP): 라이브 포트 거부 · 좌표 · 위험 조종 · 부딪힘."""

from __future__ import annotations

import io
import json

import pytest

from moai_echomind_fly.npc.box_world import BoxWorld
from moai_echomind_fly.npc.fake_world import Thing


class FakeHTTP:
    def __init__(self):
        self.calls = []
        self.bodies = {}

    def __call__(self, req, timeout=None):
        path = req.full_url.split("9412", 1)[1]
        body = json.loads(req.data) if req.data else None
        self.calls.append((req.get_method(), path, body))
        out: dict = {}
        if path.startswith("/exp/bodies") and req.get_method() == "POST":
            self.bodies[body["id"]] = {"x": body["x"], "z": body["z"],
                                       "heading_deg": body["heading_deg"], "speed": 0.0}
        elif path.startswith("/exp/near/"):
            ref = path.split("/exp/near/")[1].split("?")[0]
            others = [{"id": k, "class": "person", "size": {"radius_m": 0.35}, **v}
                      for k, v in self.bodies.items() if k != ref]
            out = {"self": self.bodies[ref], "near": others}
        elif path.startswith("/exp/steer/"):
            ref = path.split("/exp/steer/")[1]
            self.bodies[ref].update(heading_deg=body["heading_deg"], speed=body["speed"])
            blocked = ref == "hazard"
            out = {"blocked": blocked, "blocked_by": {"id": "npc"} if blocked else None,
                   "contacts": []}
        elif path == "/exp/step":
            out = {"sim_s": 0.3, "contacts": {}}
        return io.BytesIO(json.dumps(out).encode())


def test_live_port_is_refused():
    with pytest.raises(ValueError, match="라이브"):
        BoxWorld("http://127.0.0.1:9212", [9212])


def test_axes_map_and_hazard_is_steered_and_blocked_means_hit():
    http = FakeHTTP()
    w = BoxWorld("http://127.0.0.1:9412", [9212], opener=http)
    assert BoxWorld.to_world(1.0, 2.0) == (2.0, 1.0)
    w.add(Thing("npc", "resident", 1.0, 2.0))
    w.add(Thing("goal", "object", 5.0, 5.0))
    w.add(Thing("hazard", "resident", 4.0, 2.0, speed_m_s=1.0, target=(1.0, 2.0)))
    assert http.bodies["npc"]["x"] == 2.0 and http.bodies["npc"]["z"] == 1.0
    assert "goal" not in http.bodies                                     # 목표는 가상
    st = w.state("npc")
    assert (st["x"], st["y"]) == (1.0, 2.0)
    assert {n["ref"] for n in w.neighbors("npc", 50)} == {"hazard", "goal"}
    w.step(0.3)
    steer = [c for c in http.calls if c[1] == "/exp/steer/hazard"][0][2]
    assert steer["heading_deg"] == pytest.approx(180.0)                  # 우리 −x 쪽 = 세계 −z
    assert w.hit

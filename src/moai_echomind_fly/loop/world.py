"""몸 프로세스 연결 (R-3340 F-010 · 1.3-8) — FlyGym 1.x 는 파이썬 3.12 전용이라 따로 돈다.

`body/.venv/bin/python body/fly_body_server.py` 를 띄우고 JSON 한 줄씩 주고받는다.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import paths
from .teacher import Truth, bearing_deg


@dataclass
class Obs:
    t_ms: float
    pos: tuple[float, float]
    heading_rad: float
    odor_sensors: list[float]
    leg_contact: dict[str, float]
    looming: dict[str, Any]
    odor_xy: tuple[float, float]
    dead: bool
    wall_s: float
    extra: dict[str, Any] = field(default_factory=dict)   # task-v2: 몸 자세(fly_euler_rad) 등

    @property
    def food_distance_mm(self) -> float:
        return math.dist(self.pos, self.odor_xy)


def make_task(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """씨앗으로 과제를 뽑는다(냄새 원 · 다가오는 물체)."""
    w = cfg["world"]
    rng = np.random.default_rng(seed)
    d = rng.uniform(*w["odor_distance_mm"])
    b = math.radians(rng.uniform(*w["odor_bearing_deg"]))
    task: dict[str, Any] = {"odor_xy": [d * math.cos(b), d * math.sin(b)],
                            "odor_peak": w["odor_peak"], "heading0": 0.0}
    if rng.random() < w["looming_probability"]:
        side = 1 if rng.random() < 0.5 else -1
        task["looming"] = {
            "t0_ms": float(rng.uniform(*w["looming_t0_ms"])),
            "bearing_deg": side * float(rng.uniform(*w["looming_bearing_deg"])),
            "start_mm": w["looming_start_mm"], "speed_mm_s": w["looming_speed_mm_s"],
            "radius_mm": w["looming_radius_mm"], "origin_xy": None, "heading_at_t0": None,
        }
        if "looming_elevation_deg" in w:   # task-v2 — 앞선 뽑기 순서를 지키려고 맨 뒤에 뽑는다
            task["looming"]["elevation_deg"] = float(rng.uniform(*w["looming_elevation_deg"]))
    return task


class BodyClient:
    def __init__(self) -> None:
        body = paths.root() / "body"
        self.proc = subprocess.Popen(
            [str(body / ".venv" / "bin" / "python"), str(body / "fly_body_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)

    def _ask(self, req: dict[str, Any]) -> Obs:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("몸 프로세스가 답하지 않는다(죽었을 수 있다)")
        o = json.loads(line)
        if "error" in o:
            raise RuntimeError(o["error"])
        return Obs(o["t_ms"], tuple(o["pos"]), o["heading_rad"], o["odor_sensors"],
                   o["leg_contact"], o["looming"], tuple(o["odor_xy"]), o["dead"], o["wall_s"],
                   {k: o[k] for k in ("fly_euler_rad", "head_z_mm") if k in o})

    def reset(self, seed: int, task: dict[str, Any]) -> Obs:
        return self._ask({"cmd": "reset", "seed": seed, "task": task})

    def step(self, drive: tuple[float, float], ms: float) -> Obs:
        return self._ask({"cmd": "step", "drive": list(drive), "ms": ms})

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._ask({"cmd": "close"})
            except (RuntimeError, BrokenPipeError, json.JSONDecodeError):
                pass
            self.proc.wait(timeout=10)


def truth_of(obs: Obs, sensors: dict[str, int]) -> Truth:
    lo = obs.looming
    approaching = bool(lo["present"] and lo["expansion_deg_s"] > 0)
    c = sum(obs.leg_contact.values())
    return Truth(
        looming_present=bool(lo["present"] and lo["size_deg"] > 0.5),
        looming_azimuth_deg=float(lo["azimuth_deg"]), looming_size_deg=float(lo["size_deg"]),
        looming_approaching=approaching,
        odor_left=float(obs.odor_sensors[sensors["L"]]),
        odor_right=float(obs.odor_sensors[sensors["R"]]),
        odor_azimuth_deg=bearing_deg(obs.odor_xy[0] - obs.pos[0], obs.odor_xy[1] - obs.pos[1],
                                     obs.heading_rad),
        food_distance_mm=obs.food_distance_mm,
        legs_in_contact=int(sum(v > 0 for v in obs.leg_contact.values())) if c else 0,
        looming_elevation_deg=float(lo.get("elevation_deg", 0.0)),
        heading_deg=math.degrees(obs.heading_rad),
        pos_xy=(float(obs.pos[0]), float(obs.pos[1])),
        body_euler_deg=tuple(math.degrees(a) for a in obs.extra.get("fly_euler_rad", ())),
    )

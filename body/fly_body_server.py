"""몸 프로세스 (R-3340 F-010 · 1.3-8) — FlyGym 1.x(NeuroMechFly v2) · 파이썬 3.12.

뇌·Qwen(3.14 프로세스)과 표준 입출력으로 JSON 한 줄씩 주고받는다(표준 에러는 로그).
  {"cmd": "reset", "seed": 0, "task": {...}}  → 관측
  {"cmd": "step", "drive": [왼, 오], "ms": 100} → 관측   (HybridTurningController 의 좌우 하강 구동)
  {"cmd": "close"}

다가오는 물체는 **해석적**이다(Fly.exe · 06 처럼): 몸과 부딪치지 않고, 파리 머리 위치·방향에서
방위 · 각 크기 · 확대 속도만 계산한다. 냄새는 FlyGym 1.x OdorArena(거리 제곱에 반비례)다.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from dm_control.rl.control import PhysicsError
from flygym_gymnasium import Fly
from flygym_gymnasium.arena import OdorArena
from flygym_gymnasium.examples.locomotion import HybridTurningController

TIMESTEP = 1e-4
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
# 보행 제어기가 헛디딤 감지에 경골·부절 1·부절 2 접촉을 요구한다 — FlyGym 예제 배치(다리마다 6 마디)
SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACT_SENSORS = [f"{leg}{seg}" for leg in LEGS for seg in SEGMENTS]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Body:
    def __init__(self, seed: int, task: dict) -> None:
        self.task = task
        odor_xy = task["odor_xy"]
        arena = OdorArena(odor_source=np.array([[odor_xy[0], odor_xy[1], 1.0]]),
                          peak_odor_intensity=np.array([[float(task.get("odor_peak", 500.0))]]),
                          diffuse_func=lambda x: x ** -2)
        fly = Fly(enable_olfaction=True, enable_adhesion=True, draw_adhesion=False,
                  contact_sensor_placements=CONTACT_SENSORS,
                  spawn_pos=(0.0, 0.0, 0.2),
                  spawn_orientation=(0, 0, float(task.get("heading0", 0.0))))
        self.sim = HybridTurningController(fly=fly, arena=arena, timestep=TIMESTEP, seed=seed)
        self.obs, _ = self.sim.reset(seed=seed)
        self.t_ms = 0.0
        self.prev_size = None
        self.dead = False

    # 다가오는 물체 — 시작 시각에 파리 기준 방위 · 거리에서 출발해 파리 첫 자리를 향해 곧게 온다
    def looming(self, head_xy: np.ndarray, heading: float, head_z: float = 0.0) -> dict:
        lo = self.task.get("looming")
        if not lo or self.t_ms < lo["t0_ms"]:
            self.prev_size = None
            return {"present": False, "azimuth_deg": 0.0, "size_deg": 0.0, "expansion_deg_s": 0.0,
                    "distance_mm": None}
        if lo.get("origin_xy") is None:          # 출발 순간의 파리 자리 · 방향을 기준으로 잡는다
            lo["origin_xy"] = head_xy.tolist()
            lo["heading_at_t0"] = heading
            lo["origin_z"] = head_z
        b = math.radians(lo["bearing_deg"]) + lo["heading_at_t0"]
        start = np.array(lo["origin_xy"]) + lo["start_mm"] * np.array([math.cos(b), math.sin(b)])
        target = np.array(lo["origin_xy"])
        travel = lo["speed_mm_s"] * (self.t_ms - lo["t0_ms"]) / 1000.0
        dirv = (target - start) / (np.linalg.norm(target - start) + 1e-9)
        pos = start + dirv * min(travel, lo["start_mm"] + lo.get("overshoot_mm", 10.0))
        rel = pos - head_xy
        ang = math.atan2(rel[1], rel[0]) - heading
        az = math.degrees(math.atan2(math.sin(ang), math.cos(ang)))
        elev = 0.0
        if "elevation_deg" in lo:               # task-v2 — 시작 고도에서 곧게(머리 첫 높이로) 내려온다
            left = max(lo["start_mm"] - min(travel, lo["start_mm"]), 0.0)
            z = lo["origin_z"] + left * math.tan(math.radians(lo["elevation_deg"]))
            rz = z - head_z
            elev = math.degrees(math.atan2(rz, float(np.linalg.norm(rel))))
            dist = float(math.sqrt(float(np.dot(rel, rel)) + rz * rz))
        else:
            dist = float(np.linalg.norm(rel))
        size = math.degrees(2 * math.atan(lo["radius_mm"] / max(dist, lo["radius_mm"])))
        exp = 0.0 if self.prev_size is None else (size - self.prev_size) / (self.step_ms / 1000.0)
        self.prev_size = size
        return {"present": True, "azimuth_deg": az, "size_deg": size, "expansion_deg_s": exp,
                "distance_mm": dist, "elevation_deg": elev}

    def observe(self) -> dict:
        o = self.obs
        pos = np.asarray(o["fly"][0], dtype=float)
        ori = np.asarray(o["fly_orientation"], dtype=float)
        heading = math.atan2(ori[1], ori[0])
        odor = np.asarray(o["odor_intensity"], dtype=float)[0]      # [센서 4]
        forces = np.linalg.norm(np.asarray(o["contact_forces"], dtype=float), axis=1)
        per_leg = forces.reshape(len(LEGS), len(SEGMENTS)).sum(axis=1)
        return {
            "t_ms": self.t_ms, "pos": pos[:2].tolist(), "heading_rad": heading,
            "odor_sensors": odor.tolist(),
            "leg_contact": dict(zip(LEGS, per_leg.round(4).tolist())),
            "looming": self.looming(pos[:2], heading, float(pos[2])),
            "odor_xy": self.task["odor_xy"], "dead": self.dead,
            "fly_euler_rad": np.asarray(o["fly"][2], dtype=float).round(5).tolist(),
            "head_z_mm": round(float(pos[2]), 4),
        }

    def step(self, drive: list[float], ms: float) -> dict:
        self.step_ms = ms
        n = int(round(ms / 1000.0 / TIMESTEP))
        action = np.clip(np.asarray(drive, dtype=float), -0.5, 1.5)
        for _ in range(n):
            try:
                self.obs, _, term, trunc, _ = self.sim.step(action)
            except PhysicsError as err:
                log(f"[body] PhysicsError: {err}")
                self.dead = True
                break
            if term or trunc:
                break
        self.t_ms += ms
        return self.observe()


def main() -> None:
    body = None
    for line in sys.stdin:
        req = json.loads(line)
        cmd = req["cmd"]
        t0 = time.time()
        if cmd == "reset":
            if body is not None:
                body.sim.close()
            body = Body(int(req.get("seed", 0)), req["task"])
            body.step_ms = 100.0
            out = body.observe()
        elif cmd == "step":
            out = body.step(req["drive"], float(req.get("ms", 100.0)))
        elif cmd == "close":
            break
        else:
            out = {"error": f"unknown cmd {cmd}"}
        out["wall_s"] = round(time.time() - t0, 4)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

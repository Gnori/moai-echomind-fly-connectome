"""물체 인식 시험 실행기(R-5138 S0b) — NPC 가 마을을 걷고, 결정마다 참값 · 뇌 신호를 모은다.

몸: 상자 월드 · 시작 자리는 원점 둘레 무작위(씨앗) · 방향은 **실험 규칙**(곧게 걷다 막히면 돎 ·
몇 초마다 흔들기 — 뇌는 걷기 구동만) · 반사는 끔(인식만 잼) · 걷는 시험 몸 사람 몇(통과 몸).
감각: perception.scene_objects → MultiEncoder(물체마다 시각 식을 더함) · 막힘 → 앞다리 촉각.
뇌: Loop 다중 뇌 — 진짜가 몸을 몰고 섞은 배선은 **같은 감각 입력**만 받는다.
결과: runs/npc-perceive-<시각>/{features-<뇌>.npz, samples.jsonl, request.json} — 판정은
scripts/50-analysis/npc_perception_probe.py(결과 전 기준 · R-5138 S0b).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from typing import Any

import numpy as np

from .. import config, paths
from ..guard import GuardRefused, Watchdog, heavy
from ..loop.world import Obs
from .body import LEGS, ME
from .fake_world import Thing
from .perception import MultiEncoder, SceneMemory, scene_objects, truth_labels, wrap

WALK = {"walk": 0.8, "turn": "none", "turn_strength": 0.0, "stop": False, "groom": False}


def walk_policy(vectors, truth, obs, d):
    return dict(WALK), {"source": "walk"}


class PerceiveBody:
    """마을을 걷는 NPC 몸 — Loop 의 몸 자리(reset · step · close)."""

    def __init__(self, world, cfg: dict[str, Any]) -> None:
        self.w, self.cfg = world, cfg
        self.p = cfg["perceive"]
        self.s = float(cfg["scale_m_per_mm"])
        self.mem = SceneMemory()
        self.rng = np.random.default_rng(0)
        self.t_ms, self.next_turn_s, self.heading = 0.0, 0.0, 0.0
        self.created, self.blocked = False, False
        self.walkers: list[str] = [f"walker{i}" for i in range(int(self.p["walkers"]))]
        self.last_labels: dict[str, str] = {}
        self.last_objs: list[dict] = []

    def _rand_point(self, r: float) -> tuple[float, float]:
        a, d = self.rng.uniform(0, 2 * math.pi), r * math.sqrt(self.rng.uniform(0, 1))
        return d * math.cos(a), d * math.sin(a)

    def reset(self, seed: int, task: dict[str, Any]) -> Obs:
        self.rng = np.random.default_rng(seed)
        self.t_ms, self.mem = 0.0, SceneMemory()
        x, y = self._rand_point(float(self.p["start_radius_m"]))
        self.heading = float(self.rng.uniform(-180, 180))
        self.next_turn_s = float(self.p["turn_every_s"])
        if not self.created:
            self.w.reset(seed)
            self.w.add(Thing(ME, "resident", x, y, heading_deg=self.heading))
            for wref in self.walkers:
                self.w.add(Thing(wref, "walker", x + 50, y, radius_m=0.35, height_m=1.7))
            self.created = True
        else:
            self.w.place(ME, x, y, self.heading)
        for wref in self.walkers:                       # NPC 둘레에서 NPC 길을 가로지르게
            a = self.rng.uniform(0, 2 * math.pi)
            d = self.rng.uniform(*self.p["walker_start_m"])
            sx, sy = x + d * math.cos(a), y + d * math.sin(a)
            tx, ty = x - d * math.cos(a), y - d * math.sin(a)
            t = self.w.things[wref]
            t.x, t.y, t.speed_m_s, t.target = sx, sy, float(self.p["walker_speed_m_s"]), (tx, ty)
            self.w.place(wref, sx, sy, 0.0)
        return self._obs(0.0)

    def step(self, drive: tuple[float, float], bin_ms: float, reflex: bool = False) -> Obs:
        dt = bin_ms / 1000.0
        self.t_ms += bin_ms
        if self.blocked:                         # 막히면 크게 돎(실험 규칙)
            self.heading = wrap(self.heading + float(self.rng.choice([-1, 1]))
                                * float(self.rng.uniform(90, 180)))
        elif self.t_ms / 1000.0 >= self.next_turn_s:     # 몇 초마다 흔들기
            self.heading = wrap(self.heading + float(self.rng.uniform(-45, 45)))
            self.next_turn_s += float(self.p["turn_every_s"])
        left, right = drive
        mean = max(0.0, min(1.5, 0.5 * (left + right)))
        speed_mm = mean * float(self.cfg["motor"]["max_speed_mm_s"])
        self.w.steer(ME, self.heading, speed_mm * self.s, dt)
        self.w.step(dt)
        return self._obs(dt)

    def _obs(self, dt_s: float) -> Obs:
        me = self.w.state(ME)
        near = self.w.neighbors(ME, float(self.p["near_radius_m"]))
        objs = scene_objects(me, near, self.s, dt_s, self.mem, skip={ME})
        for o in objs:                                  # 걷는 시험 몸 = 사람
            if o["ref"] in self.walkers:
                o["cls"] = "person"
        self.last_objs = objs
        self.last_labels = truth_labels(objs, self.p)
        sal = max(objs, key=lambda o: o["expansion_deg_s"], default=None)
        lo = ({**sal} if sal and sal["expansion_deg_s"] > 0 else
              {"present": False, "azimuth_deg": 0.0, "size_deg": 0.0, "expansion_deg_s": 0.0,
               "elevation_deg": 0.0, "distance_mm": None})
        blocked = self.blocked = bool(me.get("blocked"))
        touch = float(self.cfg["senses"]["contact_force"]) if blocked else 0.0
        legs = {leg: (touch if leg in ("LF", "RF") else 0.0) for leg in LEGS}
        x, y = me["x"] / self.s, me["y"] / self.s
        return Obs(self.t_ms, (x, y), math.radians(me["heading_deg"]), [0.0] * 4, legs, lo,
                   (1e6, 1e6), False, time.time(), extra={"objects": objs})

    def close(self) -> None:
        pass


def main(args: argparse.Namespace) -> int:
    from .. import graph as graph_mod
    from ..loop.collect import brains_for
    from ..loop.episode import Loop
    from .run import make_world, neural_summary

    cfg = config.load("npc")
    rew = [int(x) for x in args.rewired.split(",") if x != ""]
    seeds = list(range(args.seed_start, args.seed_start + args.n))
    out = paths.runs_dir() / f"npc-perceive-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps(
        {"seeds": seeds, "brains": ["real"] + [f"rewired-s{k}" for k in rew],
         "npc": cfg["version"], "perceive": cfg["perceive"],
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False), encoding="utf-8")
    feats: dict[str, dict[str, list[np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    try:
        with heavy("npc-perceive") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 씨앗 {len(seeds)}", flush=True)
            g = graph_mod.load()
            graphs = brains_for(g, rew) if rew else {"real": g}
            if len(rew) == 1:                            # 이름을 rewired-s<K> 로 맞춤
                graphs = {"real": g, f"rewired-s{rew[0]}": graphs["rewired"]}
            loop = Loop(graphs, "real", body=False)
            c2 = copy.deepcopy(loop.cfg)
            for k, v in (cfg.get("encoder_override") or {}).items():
                c2["encoder"][k].update(v)
            c2["encoder"]["vision"].update(cfg["perceive"]["vision"])
            c2["episode"]["max_decisions"] = int(cfg["perceive"]["max_decisions"])
            c2["episode"]["sniff_decisions"] = 0
            loop.cfg = c2
            loop.encoder = MultiEncoder(loop.smap, c2, loop.loom)
            world = make_world()
            body = PerceiveBody(world, cfg)
            loop.body = body
            dog = Watchdog("scorecard")
            t = loop.table
            for i, s in enumerate(seeds):
                def on_step(step, vectors, s=s):
                    for b, vec in vectors.items():
                        per = feats.setdefault(b, {n: [] for n in vec})
                        for n, v in vec.items():
                            per[n].append(v.features(t.rate_scale_hz, t.depol_scale_mV))
                    rows.append({"seed": s, "decision": step.decision, "t_ms": step.t_ms,
                                 "labels": dict(body.last_labels),
                                 "n_objects": len(body.last_objs)})
                    near = min(body.last_objs, key=lambda o: o["distance_m"], default=None)
                    ns = neural_summary(loop, step, "real")
                    if near is not None and near["distance_m"] <= float(cfg["perceive"]["near_m"]):
                        lab = body.last_labels
                        ns["caption"] = (f"참값: 가장 가까운 {lab['person']} · {lab['side']} · "
                                         f"{near['distance_m']:.1f} m · {lab['approaching']}")
                    else:
                        ns["caption"] = "참값: 3 m 안에 없음"
                    if near is not None:
                        ns.setdefault("looming", {})["target"] = near["ref"]
                    if hasattr(world, "neural"):
                        world.neural(ME, ns)

                r = loop.episode(s, walk_policy, on_step=on_step)
                print(f"[perceive] 씨앗 {s} · 결정 {r['decisions']} · {r['wall_s']} s", flush=True)
                why = dog.check(i + 1)
                if why:
                    raise RuntimeError(f"watchdog — {why}")
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    finally:
        for b, per in feats.items():
            np.savez_compressed(out / f"features-{b}.npz",
                                **{n: np.stack(v) for n, v in per.items()})
        (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                   for r in rows), encoding="utf-8")
    print(f"[perceive] 결정 {len(rows)} · {out}", flush=True)
    return 0


def register(sub) -> None:
    p = sub.add_parser("npc-perceive",
                       help="물체 인식 시험(R-5138 S0b · 무거움) — 마을 걷기 · 참값 · 뇌 신호")
    p.add_argument("--n", type=int, default=60, help="에피소드 수(씨앗)")
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--rewired", default="0,1,2")
    p.set_defaults(func=main)

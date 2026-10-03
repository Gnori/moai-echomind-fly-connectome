"""NPC 실행기(R-5138 S0 · S1) — 몸 자리에 NPC 를 끼운 `Loop` 로 에피소드를 돌리고 판정한다.

S1(반사): 정책 = 고정 걷기(교사 · LLM 없음) · 위험이 있는 과제 씨앗만 ·
구성 = 진짜 · 재배선 sK(같은 절차) · `no_brain`(진짜 뇌를 돌리되 반사를 무시 = 그대로 걸음).
지표 = 부딪힘(과제와 같은 판정: 거리 ≤ 반지름 + hit_mm).
판정(결과 전 · I §5): 진짜 부딪힘 < 재배선 전부 · 씨앗 부호 뒤집기 p < alpha · no_brain 보다 낮음.
결과는 성적표와 같은 꼴(runs/npc-<단계>-<시각>/) — 대시보드 재사용.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from .. import config, paths
from ..guard import GuardRefused, Watchdog, heavy
from ..loop.world import make_task

WALK = {"walk": 0.8, "turn": "none", "turn_strength": 0.0, "stop": False, "groom": False}
STAND = {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True, "groom": False}


def walk_policy(vectors, truth, obs, d):
    return dict(WALK), {"source": "walk"}


def stand_policy(vectors, truth, obs, d):
    return dict(STAND), {"source": "stand"}


def hazard_seeds(start: int, n: int, cfg: dict[str, Any]) -> list[int]:
    """위험이 나오는 과제 씨앗 n 개(start 부터) — S1 은 위험 있는 장면만."""
    out, s = [], start
    while len(out) < n:
        if "looming" in make_task(s, cfg):
            out.append(s)
        s += 1
    return out


def judge_hits(results: dict[str, list[dict]]) -> dict[str, Any]:
    from ..loop.stats import STATS_RULE, sign_flip_p

    rew = sorted(c for c in results if c.startswith("rewired"))
    if "real" not in results or not rew:
        return {"rule": "npc-s1-v1", "real_beats_rewired": None}
    seeds = [e["seed"] for e in results["real"]]

    def hits(c: str) -> np.ndarray:
        by = {e["seed"]: float(bool(e["looming_hit"])) for e in results[c]}
        return np.array([by[s] for s in seeds])

    real = hits("real")
    ens = np.mean([hits(c) for c in rew], axis=0)
    p = sign_flip_p(ens - real)                     # 재배선이 더 부딪힐수록 +
    rates = {c: round(float(hits(c).mean()), 3) for c in ["real", *rew]
             + (["no_brain"] if "no_brain" in results else [])}
    beats = bool(all(rates["real"] < rates[c] for c in rew) and p < STATS_RULE["alpha"])
    below_no_brain = (rates["real"] < rates["no_brain"]) if "no_brain" in rates else None
    return {"rule": "npc-s1-v1", "hit_rate": rates, "p_sign_flip": p,
            "real_beats_rewired": beats, "real_below_no_brain": below_no_brain,
            "wording": ("초파리 배선의 반사가 섞은 배선보다 잘 피했다" if beats else
                        "섞은 배선 대조를 이기지 못했다 — '초파리 뇌로 피했다'를 쓰지 않는다")}


def main(args: argparse.Namespace) -> int:
    from .. import graph as graph_mod
    from ..loop.collect import parse_seeds
    from ..loop.episode import Loop
    from ..loop.scorecard import decision_row
    from .body import NpcBody

    task_cfg = config.load("task")
    seeds = (parse_seeds(args.seeds) if args.seeds
             else hazard_seeds(args.seed_start, args.n, task_cfg))
    configs = args.configs.split(",")
    out = paths.runs_dir() / f"npc-{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps(
        {"stage": args.stage, "seeds": seeds, "configs": configs,
         "npc": config.load("npc")["version"], "task": task_cfg["version"],
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False), encoding="utf-8")
    results: dict[str, list[dict]] = {}
    try:
        with heavy("npc-run") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 씨앗 {len(seeds)}", flush=True)
            g = graph_mod.load()
            dog = Watchdog("scorecard")
            done = 0
            for c in configs:
                brain = c if c.startswith("rewired") else "real"
                gg = g if brain == "real" else graph_mod.rewire(g, int(brain.rsplit("-s", 1)[1]))
                loop = Loop({brain: gg}, brain, body=False)
                loop.body = NpcBody(reflex_enabled=(c != "no_brain"))
                eps = []
                with (out / f"{c}.decisions.jsonl").open("w", encoding="utf-8") as dec:
                    for s in seeds:
                        def on_step(step, vectors, s=s):
                            dec.write(json.dumps(decision_row(s, step), ensure_ascii=False)
                                      + "\n")

                        pol = (stand_policy if config.load("npc").get("s1", {}).get("policy")
                               == "stand" else walk_policy)
                        r = loop.episode(s, pol, on_step=on_step)
                        r.pop("task", None)
                        eps.append(r)
                        print(f"[{c}] seed {s} · 부딪힘 {r['looming_hit']} · 최소 거리 "
                              f"{r['looming_min_mm']} · 반사 구간 {r['reflex_bins']} · "
                              f"{r['wall_s']} s", flush=True)
                        done += 1
                        why = dog.check(done)
                        if why:
                            raise RuntimeError(f"watchdog — {why}")
                (out / f"{c}.jsonl").write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps),
                    encoding="utf-8")
                results[c] = eps
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    verdict = judge_hits(results)
    (out / "card.json").write_text(json.dumps({"stage": args.stage, "seeds": seeds,
                                               "verdict": verdict}, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False))
    print(f"[npc-run] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("npc-run", help="NPC 하나를 초파리 뇌로(R-5138 · 무거움) — S1 반사")
    p.add_argument("--stage", default="s1")
    p.add_argument("--configs", default="real,rewired-s0,rewired-s1,rewired-s2,no_brain")
    p.add_argument("--seeds", default="", help="비우면 위험 있는 과제 씨앗 --n 개")
    p.add_argument("--seed-start", type=int, default=3000)
    p.add_argument("--n", type=int, default=40)
    p.set_defaults(func=main)

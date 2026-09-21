"""학습 자료 모으기 (R-3340 F-017) — 교사가 몸을 몰고, 진짜·섞은 배선 뇌 둘에 같은 감각 입력.

기록: runs/dataset-<이름>/ — features-real.npz · features-rewired.npz(채널별 [결정 수, 차원]) ·
samples.jsonl(참값 · 캡션 · 교사 행동) · episodes.jsonl · meta.json(판 번호 · 채널 표 · 대응표).
"""

from __future__ import annotations

import argparse
import json
import time

from .. import config, paths
from .. import graph as graph_mod
from ..guard import GuardRefused, heavy
from .episode import DatasetWriter, Loop, teacher_policy
from .teacher import RULES


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(args: argparse.Namespace) -> int:
    seeds = parse_seeds(args.seeds)
    out = paths.runs_dir() / f"dataset-{args.name}"
    try:
        with heavy("collect") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 에피소드 {len(seeds)}", flush=True)
            g = graph_mod.load(args.graph)
            loop = Loop({"real": g, "rewired": graph_mod.rewire(g, args.rewired_seed)}, "real")
            writer = DatasetWriter(out, loop.table)
            episodes = []
            t0 = time.time()
            try:
                for seed in seeds:
                    res = loop.episode(seed, teacher_policy,
                                       on_step=lambda s, v, seed=seed: writer.add(seed, s, v))
                    episodes.append(res)
                    print(f"[collect] seed {seed} · 결정 {res['decisions']} · "
                          f"도착 {res['reached']} · "
                          f"물체 {res['looming']} · {res['wall_s']} s · 누적 {len(writer.rows)}",
                          flush=True)
            finally:
                loop.close()
            cfgs = {n: config.load(n)["version"] for n in ("sim", "loom", "task", "channels")}
            writer.save({"graph": g.name, "rewired_seed": args.rewired_seed, "seeds": seeds,
                         "versions": cfgs, "channels": loop.table.describe(),
                         "channel_units": {n: c.units for n, c in loop.table.channels.items()},
                         "sensory_map": loop.smap.describe(), "teacher_rules": RULES,
                         "samples": len(writer.rows), "wall_s": round(time.time() - t0, 1)})
            (out / "episodes.jsonl").write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes),
                encoding="utf-8")
            print(f"[collect] 저장 {out} · 결정 {len(writer.rows)} 개", flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0

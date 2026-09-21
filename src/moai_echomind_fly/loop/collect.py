"""학습 자료 모으기 (R-3340 F-017) — 교사가 몸을 몰고, 진짜·섞은 배선 뇌 둘에 같은 감각 입력.

기록: runs/dataset-<이름>/ — features-real.npz · features-rewired.npz(채널별 [결정 수, 차원]) ·
samples.jsonl(참값 · 캡션 · 교사 행동) · episodes.jsonl · meta.json(판 번호 · 채널 표 · 대응표).
`--workers N` 이면 씨앗을 N 조각으로 나눠 자식 프로세스로 돌린다(부모가 무거운 실행 잠금을 쥔다).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .. import config, paths
from .. import graph as graph_mod
from ..guard import OWNER_ENV, GuardRefused, heavy
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


def _collect(name: str, seeds: list[int], graph_name: str, rewired_seed: int) -> int:
    out = paths.runs_dir() / f"dataset-{name}"
    g = graph_mod.load(graph_name)
    loop = Loop({"real": g, "rewired": graph_mod.rewire(g, rewired_seed)}, "real")
    writer = DatasetWriter(out, loop.table)
    episodes = []
    t0 = time.time()
    try:
        for seed in seeds:
            res = loop.episode(seed, teacher_policy,
                               on_step=lambda s, v, seed=seed: writer.add(seed, s, v))
            episodes.append(res)
            print(f"[collect {name}] seed {seed} · 결정 {res['decisions']} · "
                  f"도착 {res['reached']} · 물체 {res['looming']} · {res['wall_s']} s · "
                  f"누적 {len(writer.rows)}", flush=True)
    finally:
        loop.close()
    cfgs = {n: config.load(n)["version"] for n in ("sim", "loom", "task", "channels")}
    writer.save({"graph": g.name, "rewired_seed": rewired_seed, "seeds": seeds,
                 "versions": cfgs, "channels": loop.table.describe(),
                 "channel_units": {n: c.units for n, c in loop.table.channels.items()},
                 "sensory_map": loop.smap.describe(), "teacher_rules": RULES,
                 "samples": len(writer.rows), "wall_s": round(time.time() - t0, 1)})
    (out / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes), encoding="utf-8")
    print(f"[collect {name}] 저장 {out} · 결정 {len(writer.rows)} 개", flush=True)
    return 0


def main(args: argparse.Namespace) -> int:
    seeds = parse_seeds(args.seeds)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    if os.environ.get(OWNER_ENV) and args.workers <= 1:
        return _collect(args.name, seeds, args.graph, args.rewired_seed)   # 부모 잠금 안의 일꾼
    try:
        with heavy("collect") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 에피소드 {len(seeds)} · "
                  f"일꾼 {args.workers}", flush=True)
            if args.workers <= 1:
                return _collect(args.name, seeds, args.graph, args.rewired_seed)
            shards = [seeds[i::args.workers] for i in range(args.workers)]
            env = {**os.environ, OWNER_ENV: str(os.getpid())}
            procs = []
            for i, shard in enumerate(shards):
                cmd = [sys.executable, "-m", "moai_echomind_fly.cli", "collect",
                       "--name", f"{args.name}-w{i}", "--seeds", ",".join(map(str, shard)),
                       "--graph", args.graph, "--rewired-seed", str(args.rewired_seed),
                       "--threads", str(args.threads or 4)]
                procs.append(subprocess.Popen(cmd, env=env))
            codes = [p.wait() for p in procs]
            print(f"[collect] 일꾼 종료 코드 {codes}", flush=True)
            return 0 if all(c == 0 for c in codes) else 1
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2

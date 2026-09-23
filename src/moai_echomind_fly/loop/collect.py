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

import numpy as np

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


def brains_for(g, rewired_seeds: list[int]) -> dict:
    """진짜 + 재배선 뇌들. 씨앗 하나면 v1 이름(`rewired`), 여럿이면 `rewired-s<K>`(I-7469 §3.1)."""
    if len(rewired_seeds) == 1:
        return {"real": g, "rewired": graph_mod.rewire(g, rewired_seeds[0])}
    return {"real": g, **{f"rewired-s{k}": graph_mod.rewire(g, k) for k in rewired_seeds}}


def _collect(name: str, seeds: list[int], graph_name: str, rewired_seeds: list[int]) -> int:
    out = paths.runs_dir() / f"dataset-{name}"
    g = graph_mod.load(graph_name)
    loop = Loop(brains_for(g, rewired_seeds), "real")
    writer = DatasetWriter(out, loop.table, np.asarray(loop.smap.stim_idx))
    episodes = []
    t0 = time.time()
    try:
        for seed in seeds:
            res = loop.episode(seed, teacher_policy,
                               on_step=lambda s, v, seed=seed: writer.add(seed, s, v,
                                                                           loop.last_stim))
            episodes.append(res)
            print(f"[collect {name}] seed {seed} · 결정 {res['decisions']} · "
                  f"도착 {res['reached']} · 물체 {res['looming']} · {res['wall_s']} s · "
                  f"누적 {len(writer.rows)}", flush=True)
    finally:
        loop.close()
    cfgs = {n: config.load(n)["version"] for n in ("sim", "loom", "task", "channels", "retino")}
    writer.save({"graph": g.name, "rewired_seeds": rewired_seeds,
                 "rewired_seed": rewired_seeds[0], "seeds": seeds,
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
    rewired = parse_seeds(args.rewired_seeds) if args.rewired_seeds else [args.rewired_seed]
    if os.environ.get(OWNER_ENV) and args.workers <= 1:
        return _collect(args.name, seeds, args.graph, rewired)   # 부모 잠금 안의 일꾼
    try:
        with heavy("collect") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 에피소드 {len(seeds)} · "
                  f"일꾼 {args.workers}", flush=True)
            if args.workers <= 1:
                return _collect(args.name, seeds, args.graph, rewired)
            shards = [seeds[i::args.workers] for i in range(args.workers)]
            env = {**os.environ, OWNER_ENV: str(os.getpid())}
            procs = []
            for i, shard in enumerate(shards):
                cmd = [sys.executable, "-m", "moai_echomind_fly.cli", "collect",
                       "--name", f"{args.name}-w{i}", "--seeds", ",".join(map(str, shard)),
                       "--graph", args.graph,
                       "--rewired-seeds", ",".join(map(str, rewired)),
                       "--threads", str(args.threads or 4)]
                procs.append(subprocess.Popen(cmd, env=env))
            codes = [p.wait() for p in procs]
            print(f"[collect] 일꾼 종료 코드 {codes}", flush=True)
            return 0 if all(c == 0 for c in codes) else 1
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2


def probe(args: argparse.Namespace) -> int:
    """작은 측정(R-7469 F-002) — 에피소드 몇 개로 결정당 시간 · 최대 메모리를 재고 전체를 추정한다.

    저장하지 않는다. 기록은 runs/collect-probe-<시각>.json.
    """
    import resource

    from ..guard import process_footprint_gb

    rewired = parse_seeds(args.rewired_seeds)
    seeds = parse_seeds(args.seeds)
    out = paths.runs_dir() / f"collect-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    try:
        with heavy("collect-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 뇌 {1 + len(rewired)} · "
                  f"에피소드 {len(seeds)}", flush=True)
            t0 = time.time()
            g = graph_mod.load(args.graph)
            loop = Loop(brains_for(g, rewired), "real")
            setup_s = time.time() - t0
            rows = []
            try:
                for seed in seeds:
                    res = loop.episode(seed, teacher_policy)
                    rows.append({"seed": seed, "decisions": res["decisions"],
                                 "wall_s": res["wall_s"], "timing": res["timing"],
                                 "footprint_gb": process_footprint_gb()})
                    print(f"[probe] {rows[-1]}", flush=True)
            finally:
                loop.close()
            dec = sum(r["decisions"] for r in rows)
            per = sum(r["wall_s"] for r in rows) / max(dec, 1)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # 맥은 바이트
            rec = {"brains": 1 + len(rewired), "rewired_seeds": rewired,
                   "setup_s": round(setup_s, 1), "episodes": rows,
                   "sec_per_decision": round(per, 2), "peak_rss_gb": round(rss, 2),
                   "versions": {n: config.load(n)["version"]
                                for n in ("sim", "loom", "task", "channels", "retino")}}
            for n_ep, workers in ((300, 3), (40, 3)):
                est = n_ep * (dec / max(len(rows), 1)) * per / workers / 3600
                rec[f"estimate_{n_ep}ep_{workers}workers_h"] = round(est, 2)
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in rec.items() if k != "episodes"}, ensure_ascii=False))
            print(f"[probe] 기록 {out}")
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0

"""재주입 (R-7469 · I v2 §9.2) — 묶음 자료의 감각 입력(stim.npz)과 교사 행동으로
**몸 없이 뇌만** 다시 돌린다.

쓰임: ① 시야 대응표를 바꿔(retino-v2 · 방위) 몸을 다시 돌리지 않고 자료를 새로 만든다
② 입력을 바꾸지 않고 재주입하면 원본과 **완전히 같은** 특징이 나와야 한다
— 재주입 자체의 검증(`--verify`).

정확히 같으려면 원본 일꾼과 **같은 순서**로 에피소드를 돌려야 한다
(엔진의 포아송 난수가 에피소드를 건너 이어진다). 그래서 조각(dataset-<이름>-w<i>)마다
그 조각의 씨앗 순서대로 새 Loop 로 돈다.
시야 조율 바꾸기: 원본 자극 = 상대 세기 × 5 Hz × 게인 × 좌우 가중 × 조율 —
좌표 없는 LC 는 조율 1 이라 그 뉴런들의 발화율 ÷ 상대 세기 = (5 Hz × 게인 × 좌우 가중)이 되고,
여기에 새 조율을 곱한다.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from .. import config, datasets, paths
from .. import graph as graph_mod
from ..guard import GuardRefused, heavy
from .collect import brains_for, parse_seeds
from .decoder import IDLE
from .episode import DatasetWriter, Loop, Step
from .signals import SignalReader
from .teacher import Truth, action, caption


def retune_vision(rates: np.ndarray, smap, encoder, elevation_deg: float, azimuth_deg: float
                  ) -> np.ndarray:
    """한 구간의 자극 발화율에서 시각 부분만 새 조율로 바꾼다(다른 감각은 그대로)."""
    out = rates.copy()
    for s in ("L", "R"):
        pos, amp = smap.vision[s], smap.vision_amp[s]
        if len(pos) == 0:
            continue
        free = np.isnan(smap.vision_elev[s]) & (amp > 0)      # 원본에서 조율 1 이던 뉴런
        if not free.any():
            raise ValueError(f"{s} 눈에 조율 없는(좌표 없는) LC 가 없어 바탕 발화율을 못 얻는다")
        base = float(np.median(rates[pos][free] / amp[free]))
        out[pos] = base * amp * encoder.vision_tune(s, elevation_deg, azimuth_deg)
    return out


def _episodes(rows: list[dict]) -> dict[int, list[int]]:
    """씨앗 → 그 에피소드 결정들의 행 번호(결정 순)."""
    by: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(int(r["seed"]), []).append(i)
    for v in by.values():
        v.sort(key=lambda i: rows[i]["decision"])
    return by


def _shard_orders(pack: str, split: str, n_workers: int) -> list[list[int]]:
    """원본 일꾼별 씨앗 순서 — 조각 meta 가 있으면 그것, 없으면 seeds[i::n]."""
    meta = json.loads((datasets.store_dir() / pack / "meta.json").read_text(encoding="utf-8"))
    shards = meta.get("from_runs", {}).get(split, [])
    orders = []
    for name in shards:
        m = paths.runs_dir() / f"dataset-{name}" / "meta.json"
        if m.exists():
            orders.append([int(s) for s in json.loads(m.read_text())["seeds"]])
    if orders:
        return orders
    seeds = sorted({int(r["seed"]) for r in datasets.load(pack, split, "real")[1]})
    return [seeds[i::n_workers] for i in range(n_workers)]


def replay(pack: str, split: str, out_name: str, rewired_seeds: list[int], *,
           retune: bool, verify: int = 0, seeds: list[int] | None = None,
           log=print) -> dict[str, Any]:
    """묶음 `pack` 의 `split` 나눔을 재주입한다. `retune` 이면 지금 설정으로 시각을 바꾼다.

    `verify > 0` 이면 첫 조각의 앞 `verify` 결정만 바꾸지 않고 돌려 원본과 비교(저장 없음).
    `seeds` 가 있으면 그 에피소드만(2026-09-26 · M3 — 학습 나눔 일부).
    건너뛴 에피소드만큼 포아송 난수 흐름이 달라져 원본과 비트 단위로 같지는 않다
    (검증은 `seeds` 없이).
    """
    _, rows = datasets.load(pack, split, "real")
    stim = np.load(datasets.store_dir() / pack / split / "stim.npz")
    rates_all, stim_idx = stim["rates"], stim["stim_idx"]
    meta = json.loads((datasets.store_dir() / pack / "meta.json").read_text(encoding="utf-8"))
    g = graph_mod.load(meta["graph"])
    graphs = brains_for(g, rewired_seeds)
    loop = Loop(graphs, "real", body=False)
    if not np.array_equal(np.asarray(loop.smap.stim_idx), stim_idx):
        raise ValueError("자극 뉴런 순서가 원본과 다르다 — task · loom 판을 확인")
    by = _episodes(rows)
    if seeds is not None:
        by = {k: v for k, v in by.items() if k in set(seeds)}
    orders = _shard_orders(pack, split, 3)
    ref = ({name: datasets.load(pack, split, name)[0] for name in graphs if name in
            ("real", *datasets.rewired_brains(pack, split))} if verify else {})
    out = paths.runs_dir() / f"dataset-{out_name}"
    writer = DatasetWriter(out, loop.table, np.asarray(loop.smap.stim_idx))
    done, t0, worst = 0, time.time(), 0.0
    for order in orders:
        loop = Loop(graphs, "real", body=False)         # 조각마다 새 난수 흐름(원본과 같게)
        for seed in order:
            if seed not in by:
                continue
            for b in loop.brains.values():
                b.eng.reset()
                b.reader = SignalReader(b.eng, loop.table)
            prev = dict(IDLE)
            dn_prev = np.zeros(len(loop.decoder.dn_idx))
            for i in by[seed]:
                row = rows[i]
                dn_new = loop.decoder.target_rates(prev)
                bins = {name: [] for name in loop.brains}
                stims = []
                tr = row["truth"]
                for b in range(loop.table.bins):
                    dn = 0.5 * (dn_prev + dn_new) if b == 0 else dn_new
                    sens = rates_all[i, b].astype(np.float64)
                    if retune and tr.get("looming_present"):
                        sens = retune_vision(sens, loop.smap, loop.encoder,
                                             float(tr.get("looming_elevation_deg", 0.0)),
                                             float(tr["looming_azimuth_deg"]))
                    stims.append(sens.astype(np.float32))
                    for name, br in loop.brains.items():
                        r, d, _ = br.bin(sens, dn)
                        bins[name].append((r, d))
                dn_prev = dn_new
                prev = row["teacher_action"]   # 다음 결정의 하강 뉴런 명령(원본과 같게)
                vectors = {n: loop.brains[n].reader.assemble(bins[n]) for n in loop.brains}
                truth = Truth(**{k: (tuple(v) if isinstance(v, list) else v)
                                 for k, v in tr.items()})
                step = Step(row["decision"], row["t_ms"], truth, caption(truth), action(truth),
                            row["teacher_action"], "replay", row.get("reflex_bins", 0),
                            (0.0, 0.0), {})
                if verify:
                    f = writer.table
                    for name, vec in vectors.items():
                        if name not in ref:
                            continue
                        for ch, v in vec.items():
                            if ch not in ref[name]:   # 새 채널(channels-v3)은 원본에 없다
                                continue
                            got = v.features(f.rate_scale_hz, f.depol_scale_mV)
                            worst = max(worst, float(np.abs(got - ref[name][ch][i]).max()))
                    done += 1
                    log(f"[verify] 씨앗 {seed} 결정 {row['decision']} · 최대 차 {worst:.3g}")
                    if done >= verify:
                        return {"verified": done, "max_abs_diff": worst}
                else:
                    writer.add(seed, step, vectors, stims)
                    done += 1
            log(f"[replay {out_name}] 씨앗 {seed} 끝 · 누적 {done} · {time.time() - t0:.0f} s")
    cfgs = {n: config.load(n)["version"] for n in ("sim", "loom", "task", "channels", "retino")}
    writer.save({"graph": g.name, "rewired_seeds": rewired_seeds, "rewired_seed": rewired_seeds[0],
                 "seeds": sorted(by), "versions": cfgs, "replay_of": {"pack": pack, "split": split,
                                                                       "retune": retune},
                 "channels": loop.table.describe(),
                 "channel_units": {n: c.units for n, c in loop.table.channels.items()},
                 "sensory_map": loop.smap.describe(), "samples": len(writer.rows),
                 "teacher_rules": meta.get("teacher_rules"),      # 교사는 원본 그대로(묶기에 필요)
                 "wall_s": round(time.time() - t0, 1)})
    return {"samples": len(writer.rows), "out": str(out)}


def main(args: argparse.Namespace) -> int:
    rewired = parse_seeds(args.rewired_seeds)
    try:
        with heavy("replay") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 뇌 {1 + len(rewired)}", flush=True)
            res = replay(args.pack, args.split, args.name, rewired, retune=not args.no_retune,
                         verify=args.verify,
                         seeds=parse_seeds(args.seeds) if args.seeds else None)
            print(json.dumps(res, ensure_ascii=False))
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0


def register(sub) -> None:
    p = sub.add_parser("replay", help="묶음의 감각 입력으로 몸 없이 뇌만 다시 돌린다(무거움 · 뇌)")
    p.add_argument("--pack", default="fly-loop-v2")
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--name", required=True, help="결과 조각 이름 runs/dataset-<name>")
    p.add_argument("--rewired-seeds", default="0-4")
    p.add_argument("--no-retune", action="store_true", help="시각을 바꾸지 않는다(검증용)")
    p.add_argument("--seeds", default="", help="이 에피소드만(예: 0-59) — 비우면 전부")
    p.add_argument("--verify", type=int, default=0,
                   help="앞 N 결정만 바꾸지 않고 돌려 원본과 비교(저장 없음)")
    p.set_defaults(func=main)

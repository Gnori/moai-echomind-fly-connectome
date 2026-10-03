"""반사 진단(R-5138 S1 전) — 정면으로 다가오는 물체에 도망 뉴런(DNp01 · GF)이 켜지나.

S1 재기에서 진짜 뇌 · 뇌 없음 둘 다 부딪히고 반사 구간 0(초파리 자료도 200 중 5). 원인을 가른다:
같은 다가옴(25 mm → 0 · 25 mm/s · 반지름 1.5 mm · 정면 · 고도 0)을 인코더 설정 여럿으로 뇌에 넣고,
구간(100 ms)마다 GF 발화를 센다. 설정: 시야 모드(lr_sigmoid · retinotopic · retinotopic2) ×
증폭 상한.
몸 · 세계 없이 뇌만(Engine) — 자극 뉴런 순서는 지금 인코더 그대로.
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


def approach(start_mm: float, speed_mm_s: float, radius_mm: float, bin_ms: float):
    """정면 다가옴의 구간별 looming 칸(시각 크기 · 확대 속도 · 거리)."""
    out, d, prev = [], start_mm, None
    while d > 0:
        size = math.degrees(2 * math.atan2(radius_mm, max(d, 1e-6)))
        exp = 0.0 if prev is None else (size - prev) / (bin_ms / 1000.0)
        prev = size
        out.append({"present": True, "azimuth_deg": 0.0, "elevation_deg": 0.0,
                    "size_deg": size, "expansion_deg_s": exp, "distance_mm": d})
        d -= speed_mm_s * bin_ms / 1000.0
    return out


def run_variant(g, lif, task_cfg: dict[str, Any], mode: str, gain_max: float, seed: int,
                bin_ms: float, loom_cfg: dict[str, Any]) -> dict[str, Any]:
    from ..loop.decoder import Decoder
    from ..loop.encoder import Encoder, build_map
    from ..loop.world import Obs
    from ..protocols import LoomSpec
    from ..sim import Engine

    cfg = copy.deepcopy(task_cfg)
    cfg["encoder"]["vision"]["mode"] = mode
    cfg["encoder"]["vision"]["gain_max"] = gain_max
    spec = LoomSpec.from_config()
    smap = build_map(g, cfg, spec)
    enc = Encoder(smap, cfg, spec)
    gf = Decoder(g, cfg).escape_idx
    eng = Engine(g, lif, seed=seed)
    eng.set_poisson(smap.stim_idx, 0.0)
    frames = approach(loom_cfg["looming_start_mm"], loom_cfg["looming_speed_mm_s"],
                      loom_cfg["looming_radius_mm"], bin_ms)
    legs = {k: 0.0 for k in ("LF", "LM", "LH", "RF", "RM", "RH")}
    first, total, lc_mean = None, 0, []
    for i, lo in enumerate(frames):
        obs = Obs(i * bin_ms, (0.0, 0.0), 0.0, [0.0] * 4, legs, lo, (100.0, 0.0), False, 0.0)
        rates = enc.rates(obs)
        vis = np.concatenate([smap.vision["L"], smap.vision["R"]])
        lc_mean.append(float(rates[vis].mean()))
        eng.set_rates(rates)
        s0 = eng.step
        eng.run(bin_ms)
        _, n = eng.spikes_since(s0)
        k = int(np.isin(n, gf).sum())
        total += k
        if k and first is None:
            first = {"bin": i, "distance_mm": round(lo["distance_mm"], 2)}
    return {"mode": mode, "gain_max": gain_max, "seed": seed, "gf_spikes": total,
            "first_gf": first, "bins": len(frames),
            "lc_rate_hz_last3": [round(x, 2) for x in lc_mean[-3:]]}


def main(args: argparse.Namespace) -> int:
    from .. import graph as graph_mod
    from ..guard import GuardRefused, heavy
    from ..sim import LIFParams

    task_cfg = config.load("task")
    ch = config.load("channels")
    bin_ms = float(ch["window_ms"]) / int(ch["bins"])
    rows = []
    try:
        with heavy("reflex-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load()
            lif = LIFParams.from_config()
            for mode in args.modes.split(","):
                for gm in [float(x) for x in args.gains.split(",")]:
                    for seed in range(args.seeds):
                        r = run_variant(g, lif, task_cfg, mode, gm, seed, bin_ms,
                                        task_cfg["world"])
                        rows.append(r)
                        print(f"[reflex-probe] {mode:13} 상한 {gm:5} 씨앗 {seed} · "
                              f"GF {r['gf_spikes']}"
                              f" · 첫 GF {r['first_gf']} · LC 마지막 {r['lc_rate_hz_last3']}",
                              flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"reflex-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"bin_ms": bin_ms, "task": task_cfg["version"], "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reflex-probe] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("reflex-probe", help="반사 진단 — 정면 다가옴에 GF 가 켜지나(무거움 · 짧음)")
    p.add_argument("--modes", default="lr_sigmoid,retinotopic,retinotopic2")
    p.add_argument("--gains", default="3,10")
    p.add_argument("--seeds", type=int, default=2)
    p.set_defaults(func=main)

"""V0 — 분석식 광류 → T4/T5 → 뇌(R-1177 · 결과 전 잠금 config/eyeworld.yaml v0 rule v0-a).

팔: 진짜 대응표 · 자리 섞기 5 벌(진짜 배선) · 재배선 s0~s2(진짜 대응표) · 씨앗 10.
자극 여섯(회전 cw · ccw · 다가옴 왼 · 오 · 접선 왼 · 오) · 300 ms · 10 ms 마다 발화율.
읽는 것: DNa02 · LPLC2 · GF · LC4 · HS · VS 좌우.
사용: python scripts/60-eyeworld/v0_flow.py [--seeds 0] [--maps 1] [--brains real] [--no-judge]
     → runs/v0-flow-<시각>/{rows.jsonl, meta.json, judge.json}
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye import flow
from moai_echomind_fly.eye.geometry import Positions, t4t5_positions
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams

GROUPS = {
    "DNa02": ["DNa02"],
    "LPLC2": ["LPLC2"],
    "GF": ["DNp01"],
    "LC4": ["LC4"],
    "HS": ["HSE", "HSN", "HSS"],
    "VS": ["VS"],
}
STIMS = ["rot-cw", "rot-ccw", "loom-left", "loom-right", "tan-left", "tan-right"]


def schedule(stim: str, pos: Positions, v0: dict) -> list[np.ndarray]:
    """자극 하나의 구간별 발화율(뉴런마다) 목록."""
    n_bins = int(round(v0["window_ms"] / v0["bin_ms"]))
    r_max, v_ref = float(v0["r_max_hz"]), float(v0["v_ref_deg_s"])
    st = v0["stimuli"]
    if stim.startswith("rot"):
        m = flow.rotation_field(pos, float(st["rotation"]["speed_deg_s"]), stim.split("-")[1])
        r = flow.rates(pos, *m, r_max, v_ref)
        return [r] * n_bins
    lm = st["loom"]
    center = tuple(lm["centers_deg"][stim.split("-")[1]])
    r0, r1 = map(float, lm["radius_deg"])
    drdt = (r1 - r0) / (float(v0["window_ms"]) / 1000.0)
    out = []
    for b in range(n_bins):
        rad = r0 + (r1 - r0) * (b + 0.5) / n_bins
        loom = flow.rates(
            pos, *flow.loom_field(pos, center, rad, drdt, float(lm["edge_sigma_deg"])), r_max, v_ref
        )
        if stim.startswith("loom"):
            out.append(loom)
        else:
            tan = flow.rates(
                pos,
                *flow.loom_field(
                    pos, center, rad, drdt, float(lm["edge_sigma_deg"]), tangential=True
                ),
                r_max,
                v_ref,
            )
            out.append(flow.match_totals(pos, loom, tan, r_max))
    return out


def judge(rows: list[dict], v0: dict) -> dict:
    ps = v0["pass"]

    def vals(arm: str, stim: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["arm"] == arm and r["stim"] == stim], float)

    def lplc2_loom(arm: str, kind: str = "loom") -> float:
        return float(
            sum(
                np.mean(vals(arm, f"{kind}-{s}", "LPLC2_L") + vals(arm, f"{kind}-{s}", "LPLC2_R"))
                for s in ("left", "right")
            )
        )

    dcw = vals("real", "rot-cw", "DNa02_R") - vals("real", "rot-cw", "DNa02_L")
    dccw = vals("real", "rot-ccw", "DNa02_R") - vals("real", "rot-ccw", "DNa02_L")
    flips = int(np.sum(np.sign(dcw) * np.sign(dccw) < 0))
    real_abs = float(np.mean(np.abs(np.concatenate([dcw, dccw]))))
    rew = sorted({r["arm"] for r in rows if r["arm"].startswith("rewired")})
    shuf = sorted({r["arm"] for r in rows if r["arm"].startswith("shuf")})
    rew_abs = {
        a: float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            vals(a, s, "DNa02_R") - vals(a, s, "DNa02_L")
                            for s in ("rot-cw", "rot-ccw")
                        ]
                    )
                )
            )
        )
        for a in rew
    }
    c1 = bool(
        flips >= int(ps["c1_rotation_flip_min"])
        and all(
            v == 0 or v < float(ps["c1_rewired_max_ratio"]) * real_abs for v in rew_abs.values()
        )
    )
    real_loom = lplc2_loom("real")
    per_stim = {
        s: float(
            np.mean(vals("real", f"loom-{s}", "LPLC2_L") + vals("real", f"loom-{s}", "LPLC2_R"))
        )
        for s in ("left", "right")
    }
    insufficient = any(v < float(ps["insufficient_lplc2"]) for v in per_stim.values())
    shuf_loom = {a: lplc2_loom(a) for a in shuf}
    tan = lplc2_loom("real", "tan")
    ipsi = {
        s: float(
            np.mean(
                vals("real", f"loom-{s}", f"LPLC2_{s[0].upper()}")
                - vals("real", f"loom-{s}", f"LPLC2_{'R' if s == 'left' else 'L'}")
            )
        )
        for s in ("left", "right")
    }
    med = float(np.median(list(shuf_loom.values()))) if shuf_loom else float("nan")
    c2_parts = {
        "above_all_shuffles": bool(shuf_loom) and all(real_loom > v for v in shuf_loom.values()),
        "ratio_vs_shuffle_median": real_loom / med if med > 0 else float("inf"),
        "ratio_vs_tangential": real_loom / tan if tan > 0 else float("inf"),
        "ipsi_both": all(v > 0 for v in ipsi.values()),
    }
    c2 = (
        not insufficient
        and c2_parts["above_all_shuffles"]
        and c2_parts["ratio_vs_shuffle_median"] >= float(ps["c2_shuffle_median_ratio"])
        and c2_parts["ratio_vs_tangential"] >= float(ps["c2_vs_tangential_ratio"])
        and c2_parts["ipsi_both"]
    )
    rew_loom = {a: lplc2_loom(a) for a in rew}
    c3 = bool(rew_loom) and all(v < real_loom for v in rew_loom.values())
    return {
        "rule": v0["rule"],
        "C1": {
            "pass": c1,
            "flips": flips,
            "n": len(dcw),
            "dna02_RminusL_cw": dcw.tolist(),
            "dna02_RminusL_ccw": dccw.tolist(),
            "real_mean_abs": real_abs,
            "rewired_mean_abs": rew_abs,
            "syndirectional_cw_right": bool(np.mean(dcw) > 0),
        },
        "C2": {
            "pass": bool(c2),
            "insufficient": insufficient,
            "real_loom_lplc2": real_loom,
            "per_stim": per_stim,
            "shuffles": shuf_loom,
            "shuffle_median": med,
            "tangential": tan,
            "ipsi_L_minus_R": ipsi,
            **c2_parts,
        },
        "C3": {"pass": c3, "rewired_loom_lplc2": rew_loom},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="쉼표 목록 · 기본 설정 seeds")
    ap.add_argument("--maps", type=int, default=None, help="자리 섞기 벌 수 · 기본 설정")
    ap.add_argument("--brains", default=None, help="쉼표 목록 · 기본 real + 설정 rewired")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    v0 = config.load("eyeworld")["v0"]
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [int(x) for x in v0["seeds"]]
    n_maps = int(v0["arms"]["shuffled_maps"]["n"]) if a.maps is None else a.maps
    brains = a.brains.split(",") if a.brains else ["real", *v0["arms"]["rewired"]]
    out = paths.runs_dir() / f"v0-flow-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    rows: list[dict] = []
    t_start = time.time()
    with heavy("v0-flow") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base)
        groups = {
            f"{k}_{s}": np.flatnonzero(np.isin(typ, v) & (side == s))
            for k, v in GROUPS.items()
            for s in ("L", "R")
        }
        real = t4t5_positions(base)
        s0 = int(v0["arms"]["shuffled_maps"]["seed0"])
        maps = [("real", real)] + [(f"shuf{k}", real.shuffled(s0 + k)) for k in range(n_maps)]
        sched = {(m, s): schedule(s, p, v0) for m, p in maps for s in STIMS}
        print(
            f"[v0] T4/T5 {len(real)} · 지도 {len(maps)} · 자극 {len(STIMS)}"
            f" · 씨앗 {seeds} · 뇌 {brains}",
            flush=True,
        )
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "rule": v0["rule"],
                    "seeds": seeds,
                    "maps": [m for m, _ in maps],
                    "brains": brains,
                    "stims": STIMS,
                    "n_t4t5": len(real),
                    "graph": base.name,
                    "lif": lif.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        f = (out / "rows.jsonl").open("w", encoding="utf-8")
        for brain in brains:
            g = graph_mod.brain_graph(base, brain)
            arm_maps = maps if brain == "real" else maps[:1]
            for seed in seeds:
                eng = Engine(g, lif, device="cpu", seed=seed)
                eng.set_poisson(real.idx, np.zeros(len(real)))
                for m, _ in arm_maps:
                    for stim in STIMS:
                        t0 = time.time()
                        eng.reset()
                        for r in sched[(m, stim)]:
                            eng.set_rates(r)
                            eng.run(float(v0["bin_ms"]))
                        _, neurons = eng.spikes()
                        cnt = np.bincount(neurons, minlength=g.n)
                        arm = m if brain == "real" else brain
                        row = {
                            "arm": arm,
                            "brain": brain,
                            "map": m,
                            "stim": stim,
                            "seed": seed,
                            "input_hz_sum": float(np.mean([r.sum() for r in sched[(m, stim)]])),
                            "sec": round(time.time() - t0, 2),
                            **{k: int(cnt[ix].sum()) for k, ix in groups.items()},
                        }
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        print(
                            f"[v0] {arm:10} {stim:10} 씨앗 {seed}"
                            f" · DNa02 L/R {row['DNa02_L']}/{row['DNa02_R']}"
                            f" · LPLC2 L/R {row['LPLC2_L']}/{row['LPLC2_R']}"
                            f" · GF {row['GF_L']}/{row['GF_R']} · "
                            f"HS {row['HS_L']}/{row['HS_R']} · {row['sec']} s",
                            flush=True,
                        )
        f.close()
    print(f"[v0] 실행 {len(rows)} · {time.time() - t_start:.0f} s", flush=True)
    if not a.no_judge:
        res = judge(rows, v0)
        (out / "judge.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[v0] 판정 C1 {res['C1']['pass']} · C2 {res['C2']['pass']} · C3 {res['C3']['pass']}",
            flush=True,
        )
    print(f"[v0] {out}")


if __name__ == "__main__":
    main()

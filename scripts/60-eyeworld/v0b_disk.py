"""V0b — 어두운 원판 → [T4/T5 + 소엽 기둥 종류] → LC4 · LPLC2 · GF.

R-1177 · config/eyeworld.yaml v0b.

설정에 "미정" 칸이 하나라도 있으면 거부한다(문헌 값으로 결과 전에 잠근 뒤에만 돈다).
팔: 진짜 대응표 · 자리 섞기 5 벌(T4/T5 · 소엽 종류 모두) · 재배선 s0~s2 · 씨앗 10.
자극 여섯(다가옴 · 물러남 · 어두워지기만 × 왼 · 오) · 300 ms · 10 ms 마다 발화율.
사용: python scripts/60-eyeworld/v0b_disk.py [--seeds 0] [--maps 1] [--brains real] [--no-judge]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye import disk, flow
from moai_echomind_fly.eye.geometry import Positions, t4t5_positions, type_positions
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams

READ = {"LC4": ["LC4"], "LPLC2": ["LPLC2"], "GF": ["DNp01"], "DNp02": ["DNp02"], "DNa02": ["DNa02"]}
KINDS = ["loom", "recede", "dim"]


def pending(x, path: str = "") -> list[str]:
    """설정 안의 '미정' 칸 경로."""
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in pending(v, f"{path}.{k}" if path else str(k))]
    return [path] if x == "미정" else []


def concat(a: Positions, b: Positions) -> Positions:
    return Positions(
        *(
            np.concatenate([getattr(a, f), getattr(b, f)])
            for f in ("idx", "kind", "sub", "side", "h", "v", "az", "el")
        )
    )


def schedule(kind: str, where: str, pos: Positions, n_t45: int, cfg: dict) -> list[np.ndarray]:
    """자극 하나의 구간별 발화율 — 앞 n_t45 개는 T4/T5, 나머지는 소엽 종류."""
    dk, n_bins = cfg["disk"], int(round(cfg["window_ms"] / cfg["bin_ms"]))
    bin_s, r_max = float(cfg["bin_ms"]) / 1000.0, float(cfg["r_max_hz"])
    center = tuple(dk["centers_deg"][where])
    r0, r1 = map(float, dk["radius_deg"])
    radii, contrasts = disk.disk_course(kind, r0, r1, float(dk["contrast"]), n_bins)
    t45 = Positions(
        *(getattr(pos, f)[:n_t45] for f in ("idx", "kind", "sub", "side", "h", "v", "az", "el"))
    )
    lob = Positions(
        *(getattr(pos, f)[n_t45:] for f in ("idx", "kind", "sub", "side", "h", "v", "az", "el"))
    )
    d_lob = disk.angular_distance(lob, center)
    lob_rates = np.zeros((n_bins, len(lob)))
    for t, p in cfg["lobula"].items():
        m = lob.kind == t
        rs = disk.lobula_rates(
            d_lob[m],
            radii,
            contrasts,
            bin_s,
            p["polarity"],
            p["mode"],
            float(p["rf_sigma_deg"]),
            float(p["ref"]),
            r_max,
        )
        lob_rates[:, m] = np.array(rs)
    drdt = {"loom": (r1 - r0), "recede": -(r1 - r0), "dim": 0.0}[kind] / (n_bins * bin_s)
    polarity = "off" if kind == "loom" else "on"  # 다가옴 = 어두워지는 가장자리 · 물러남 = 밝아짐
    out = []
    for b in range(n_bins):
        m = flow.loom_field(t45, center, float(radii[b]), drdt, float(dk["edge_sigma_deg"]))
        r = flow.rates(t45, *m, r_max, float(cfg["t4t5"]["v_ref_deg_s"]))
        if cfg["t4t5"]["split_on_off"]:
            keep = "T5" if polarity == "off" else "T4"
            r = np.where(t45.kind == keep, r, 0.0)
        out.append(np.concatenate([r, lob_rates[b]]))
    return out


def judge(rows: list[dict], cfg: dict) -> dict:
    ps = cfg["pass"]
    res: dict = {"rule": cfg["rule"]}

    def mean(arm: str, stim: str, key: str) -> float:
        v = [r[key] for r in rows if r["arm"] == arm and r["stim"] == stim]
        return float(np.mean(v)) if v else float("nan")

    arms = sorted({r["arm"] for r in rows})
    for ro in ps["readouts"]:

        def total(arm: str, kind: str = "loom", ro: str = ro) -> float:
            return sum(
                mean(arm, f"{kind}-{w}", f"{ro}_L") + mean(arm, f"{kind}-{w}", f"{ro}_R")
                for w in ("left", "right")
            )

        real = total("real")
        per = {
            w: mean("real", f"loom-{w}", f"{ro}_L") + mean("real", f"loom-{w}", f"{ro}_R")
            for w in ("left", "right")
        }
        ipsi = {
            w: mean("real", f"loom-{w}", f"{ro}_{w[0].upper()}")
            - mean("real", f"loom-{w}", f"{ro}_{'R' if w == 'left' else 'L'}")
            for w in ("left", "right")
        }
        shuf = {a: total(a) for a in arms if a.startswith("shuf")}
        rew = {a: total(a) for a in arms if a.startswith("rewired")}
        med = float(np.median(list(shuf.values()))) if shuf else float("nan")
        dim, recede = total("real", "dim"), total("real", "recede")
        insufficient = any(v < float(ps["insufficient"]) for v in per.values())
        parts = {
            "above_all_shuffles": bool(shuf) and all(real > v for v in shuf.values()),
            "ratio_vs_shuffle_median": real / med if med > 0 else float("inf"),
            "ratio_vs_dim": real / dim if dim > 0 else float("inf"),
            "ipsi_both": all(v > 0 for v in ipsi.values()),
            "rewired_below": bool(rew) and all(v < real for v in rew.values()),
        }
        ok = (
            not insufficient
            and parts["above_all_shuffles"]
            and parts["ipsi_both"]
            and parts["rewired_below"]
            and parts["ratio_vs_shuffle_median"] >= float(ps["shuffle_median_ratio"])
            and parts["ratio_vs_dim"] >= float(ps["vs_dim_ratio"])
        )
        res[ro] = {
            "pass": bool(ok),
            "insufficient": insufficient,
            "real_loom": real,
            "per_stim": per,
            "ipsi": ipsi,
            "shuffles": shuf,
            "shuffle_median": med,
            "rewired": rew,
            "dim": dim,
            "recede": recede,
            **parts,
        }
    res["GF_loom_real"] = {
        w: mean("real", f"loom-{w}", f"GF_{w[0].upper()}") for w in ("left", "right")
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--maps", type=int, default=None)
    ap.add_argument("--brains", default=None)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    cfg = config.load("eyeworld")["v0b"]
    miss = pending(cfg)
    if miss:
        print(
            f"[v0b] 거부 — 설정에 '미정' {len(miss)} 칸: {miss[:8]} …"
            " (문헌 값으로 결과 전 잠금 먼저)"
        )
        sys.exit(2)
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [int(x) for x in cfg["seeds"]]
    n_maps = int(cfg["arms"]["shuffled_maps"]["n"]) if a.maps is None else a.maps
    brains = a.brains.split(",") if a.brains else ["real", *cfg["arms"]["rewired"]]
    out = paths.runs_dir() / f"v0b-disk-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    rows: list[dict] = []
    stims = [(k, w) for k in KINDS for w in ("left", "right")]
    with heavy("v0b-disk") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base)
        groups = {
            f"{k}_{s}": np.flatnonzero(np.isin(typ, v) & (side == s))
            for k, v in READ.items()
            for s in ("L", "R")
        }
        t45 = t4t5_positions(base)
        real = concat(t45, type_positions(base, list(cfg["lobula"].keys())))
        s0 = int(cfg["arms"]["shuffled_maps"]["seed0"])
        maps = [("real", real)] + [(f"shuf{k}", real.shuffled(s0 + k)) for k in range(n_maps)]
        sched = {(m, k, w): schedule(k, w, p, len(t45), cfg) for m, p in maps for k, w in stims}
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "rule": cfg["rule"],
                    "seeds": seeds,
                    "maps": [m for m, _ in maps],
                    "brains": brains,
                    "n_t4t5": len(t45),
                    "n_lobula": len(real) - len(t45),
                    "lobula": cfg["lobula"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for brain in brains:
                g = graph_mod.brain_graph(base, brain)
                for seed in seeds:
                    eng = Engine(g, lif, device="cpu", seed=seed)
                    eng.set_poisson(real.idx, np.zeros(len(real)))
                    for m, _ in maps if brain == "real" else maps[:1]:
                        for k, w in stims:
                            eng.reset()
                            for r in sched[(m, k, w)]:
                                eng.set_rates(r)
                                eng.run(float(cfg["bin_ms"]))
                            _, neurons = eng.spikes()
                            cnt = np.bincount(neurons, minlength=g.n)
                            row = {
                                "arm": m if brain == "real" else brain,
                                "stim": f"{k}-{w}",
                                "seed": seed,
                                **{n: int(cnt[ix].sum()) for n, ix in groups.items()},
                            }
                            rows.append(row)
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()
                            print(
                                f"[v0b] {row['arm']:10} {row['stim']:12} 씨앗 {seed}"
                                f" · LC4 {row['LC4_L']}/{row['LC4_R']}"
                                f" · LPLC2 {row['LPLC2_L']}/{row['LPLC2_R']}"
                                f" · GF {row['GF_L']}/{row['GF_R']}",
                                flush=True,
                            )
    if not a.no_judge:
        res = judge(rows, cfg)
        (out / "judge.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "[v0b] 판정 " + " · ".join(f"{ro} {res[ro]['pass']}" for ro in cfg["pass"]["readouts"])
        )
    print(f"[v0b] {out}")


if __name__ == "__main__":
    main()

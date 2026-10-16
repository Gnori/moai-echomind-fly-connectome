"""V0b — 어두운 원판 → [T5 + 측정된 소엽 기둥 종류] → LC4(주 판정) · LPLC2 · GF(보고).

R-1177 · config/eyeworld.yaml v0b(rule v0b-a · 결과 전 잠금 · 연구 노트 17 §10.4).
설정에 "미정" 칸이 하나라도 있으면 거부한다.
팔: main(T5 · Tm2 · Tm4 · T2) 진짜 대응표 · 자리 섞기 5 벌 · 재배선 s0~s2
/ no_t2 · explore(진짜 · 보고).
자극 여섯(다가옴 · 어두워지기만 · 접선 × 왼 · 오) · 300 ms · 10 ms 마다 발화율 · 씨앗 10.
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

READ = {
    "LC4": ["LC4"],
    "LPLC2": ["LPLC2"],
    "GF": ["DNp01"],
    "DNp02": ["DNp02"],
    "DNp11": ["DNp11"],
    "DNa02": ["DNa02"],
}
FIELDS = ("idx", "kind", "sub", "side", "h", "v", "az", "el")


def pending(x, path: str = "") -> list[str]:
    """설정 안의 '미정' 칸 경로."""
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in pending(v, f"{path}.{k}" if path else str(k))]
    return [path] if x == "미정" else []


def concat(*ps: Positions) -> Positions:
    return Positions(*(np.concatenate([getattr(p, f) for p in ps]) for f in FIELDS))


def pick(p: Positions, m: np.ndarray) -> Positions:
    return Positions(*(getattr(p, f)[m] for f in FIELDS))


def schedule(kind: str, where: str, pos: Positions, cfg: dict) -> list[np.ndarray]:
    """자극 하나의 구간별 발화율(pos 순서) — T5 는 가장자리 움직임 · 소엽 종류는 밝기 변화."""
    dk, n_bins = cfg["disk"], int(round(cfg["window_ms"] / cfg["bin_ms"]))
    bin_s, r_max = float(cfg["bin_ms"]) / 1000.0, float(cfg["r_max_hz"])
    v_ref = float(cfg["t5"]["v_ref_deg_s"])
    center = tuple(dk["centers_deg"][where])
    r0, r1 = map(float, dk["radius_deg"])
    lum_kind = "dim" if kind == "dim" else "loom"
    radii, contrasts = disk.disk_course(lum_kind, r0, r1, float(dk["contrast"]), n_bins)
    t5 = pos.kind == "T5"
    t4 = pos.kind == "T4"
    lob = ~(t5 | t4)
    rates = np.zeros((n_bins, len(pos)))
    params = {**cfg["lobula"], **cfg["explore"]}
    lp = pick(pos, lob)
    d = disk.angular_distance(lp, center)
    lob_r = np.zeros((n_bins, len(lp)))
    for t in sorted(set(lp.kind)):
        p = params[t]
        m = lp.kind == t
        sig = float(p["rf_sigma_deg"])
        ref = disk.edge_ref(float(dk["contrast"]), v_ref, sig)
        lob_r[:, m] = np.array(
            disk.lobula_rates(
                d[m],
                radii,
                contrasts,
                bin_s,
                p["polarity"],
                p["mode"],
                sig,
                ref,
                r_max,
                float(p["tau_ms"]),
            )
        )
    rates[:, lob] = lob_r
    if kind != "dim":
        tp = pick(pos, t5)
        drdt = (r1 - r0) / (n_bins * bin_s)
        for b in range(n_bins):
            rad, sig = float(radii[b]), float(dk["edge_sigma_deg"])
            loom = flow.rates(tp, *flow.loom_field(tp, center, rad, drdt, sig), r_max, v_ref)
            if kind == "tan":
                tan = flow.rates(
                    tp, *flow.loom_field(tp, center, rad, drdt, sig, tangential=True), r_max, v_ref
                )
                loom = flow.match_totals(tp, loom, tan, r_max)
            rates[b, t5] = loom
    # T4 는 어두운 가장자리라 0(설정 t4_zero) — pos 에 T4 가 있어도 0 그대로
    return list(rates)


def judge(rows: list[dict], cfg: dict) -> dict:
    ps = cfg["pass"]
    ro = ps["readout"]

    def vals(arm: str, stim: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["arm"] == arm and r["stim"] == stim], float)

    def total(arm: str, kind: str = "loom", key: str = ro) -> float:
        return float(
            sum(
                np.mean(vals(arm, f"{kind}-{w}", f"{key}_L") + vals(arm, f"{kind}-{w}", f"{key}_R"))
                for w in ("left", "right")
            )
        )

    arms = sorted({r["arm"] for r in rows})
    real = total("main")
    per = {
        w: float(
            np.mean(vals("main", f"loom-{w}", f"{ro}_L") + vals("main", f"loom-{w}", f"{ro}_R"))
        )
        for w in ("left", "right")
    }
    insufficient = any(v < float(ps["insufficient"]) for v in per.values())
    shuf = {a: total(a) for a in arms if a.startswith("shuf")}
    med = float(np.median(list(shuf.values()))) if shuf else float("nan")
    e1 = (
        not insufficient
        and bool(shuf)
        and all(real > v for v in shuf.values())
        and (real / med if med > 0 else float("inf")) >= float(ps["e1_shuffle_median_ratio"])
    )
    rew = {a: total(a) for a in arms if a.startswith("rewired")}
    e2 = bool(rew) and all(v < real for v in rew.values())
    lft = vals("main", "loom-left", f"{ro}_L") - vals("main", "loom-left", f"{ro}_R")
    rgt = vals("main", "loom-right", f"{ro}_R") - vals("main", "loom-right", f"{ro}_L")
    ipsi_seeds = int(np.sum((lft > 0) & (rgt > 0)))
    e3 = ipsi_seeds >= int(ps["e3_ipsi_min"])
    report = {
        k: {
            "main_loom": total("main", key=k),
            "main_dim": total("main", "dim", k),
            "main_tan": total("main", "tan", k),
            "no_t2_loom": total("no_t2", key=k),
            "explore_loom": total("explore", key=k),
            "rewired_loom": {a: total(a, key=k) for a in rew},
        }
        for k in ("LC4", "LPLC2", "GF", "DNp02", "DNp11")
    }
    return {
        "rule": cfg["rule"],
        "readout": ro,
        "pass": bool(e1 and e2 and e3),
        "E1": {
            "pass": bool(e1),
            "insufficient": insufficient,
            "real_loom": real,
            "per_stim": per,
            "shuffles": shuf,
            "shuffle_median": med,
            "ratio_vs_median": real / med if med > 0 else None,
        },
        "E2": {"pass": e2, "rewired": rew},
        "E3": {
            "pass": e3,
            "ipsi_seeds": ipsi_seeds,
            "left_ipsi_minus_contra": lft.tolist(),
            "right_ipsi_minus_contra": rgt.tolist(),
        },
        "report": report,
    }


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
    kinds = list(cfg["disk"]["kinds"])
    stims = [(k, w) for k in kinds for w in ("left", "right")]
    out = paths.runs_dir() / f"v0b-disk-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    rows: list[dict] = []
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
        t5 = t4t5_positions(base, ["T5a", "T5b", "T5c", "T5d"])

        def arm_pos(names: list[str]) -> Positions:
            lob = [n for n in names if n != "T5"]
            return concat(t5, type_positions(base, lob)) if lob else t5

        main_pos = arm_pos(cfg["arms"]["main"])
        s0 = int(cfg["arms"]["shuffled_maps"]["seed0"])
        real_arms = [
            ("main", main_pos),
            ("no_t2", arm_pos(cfg["arms"]["no_t2"])),
            ("explore", arm_pos(cfg["arms"]["explore"])),
        ]
        real_arms += [(f"shuf{k}", main_pos.shuffled(s0 + k)) for k in range(n_maps)]
        sched = {(m, k, w): schedule(k, w, p, cfg) for m, p in real_arms for k, w in stims}
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "rule": cfg["rule"],
                    "seeds": seeds,
                    "arms": [m for m, _ in real_arms],
                    "brains": brains,
                    "stims": [f"{k}-{w}" for k, w in stims],
                    "n": {m: len(p) for m, p in real_arms},
                    "lobula": cfg["lobula"],
                    "explore": cfg["explore"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for brain in brains:
                g = graph_mod.brain_graph(base, brain)
                arms = real_arms if brain == "real" else [("main", main_pos)]
                for seed in seeds:
                    eng = Engine(g, lif, device="cpu", seed=seed)
                    for m, p in arms:
                        eng.set_poisson(p.idx, np.zeros(len(p)))
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
                                f"[v0b] {row['arm']:10} {row['stim']:11} 씨앗 {seed}"
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
            f"[v0b] 판정 {res['pass']} · E1 {res['E1']['pass']} · E2 {res['E2']['pass']}"
            f" · E3 {res['E3']['pass']}",
            flush=True,
        )
    print(f"[v0b] {out}")


if __name__ == "__main__":
    main()

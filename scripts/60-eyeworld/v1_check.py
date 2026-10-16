"""V1 — 렌더러 그림 → 앞단 v2 → 뇌(R-1177 · 열린 고리).

결과 전 잠금: config/eyeworld.yaml v1 rule v1-a.
① 그리기: 조건 넷(몸 돎 왼 · 오 · 사람 다가옴 왼 · 오)마다 세계를 대본대로 움직이며
   두 눈 큐브맵을 그려 저장.
② 앞단: 조건마다 FrontendV2 로 발화율 순서(입구 뉴런 20,695).
③ 뇌: 진짜 대응표 · 자리 섞기 5 벌(입구 뉴런 자리만 종류 · 아형 × 눈 안에서 섞음) ·
   재배선 s0~s2 · 씨앗 10.
④ 판정: R1(돎 → DNa02 부호) · A1~A3(다가옴 → LC4 자리 · 배선 · 같은 쪽).
사용: python scripts/60-eyeworld/v1_check.py [--frames <폴더>] [--seeds 0] [--maps 1]
     [--brains real]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye.frontend import FrontendV2
from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.render import Renderer
from moai_echomind_fly.eyeworld.world import Command, Entity, body_dir
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams

READ = {
    "DNa02": ["DNa02"],
    "LC4": ["LC4"],
    "LPLC2": ["LPLC2"],
    "GF": ["DNp01"],
    "DNp02": ["DNp02"],
    "HS": ["HSE", "HSN", "HSS"],
    "VS": ["VS"],
}
CONDS = ["rot-left", "rot-right", "app-left", "app-right"]


def build_world(v1: dict, cond: str, seed: int):
    """조건마다 새 세계 — 사람 개체는 늘 넣되(렌더러가 처음에 싣게) 다가옴이 아니면 멀리."""
    world, scene, _ = scene_mod.load(v1["scene"], seed=seed)
    ap = v1["approach"]
    scene["assets"][ap["asset"]] = {
        "path": ap["asset_path"],
        "license": ap["license"],
        "source": ap["source"],
    }
    scene_mod.verify_assets({"assets": {ap["asset"]: scene["assets"][ap["asset"]]}})
    fly = world.ents["fly"]
    if cond.startswith("app"):
        rig = world.eye_rig("fly")
        mid = (np.asarray(rig[0]["pos"]) + np.asarray(rig[1]["pos"])) / 2
        d = body_dir(
            fly.yaw, np.array(float(ap["bearings_deg"][cond.split("-")[1]])), np.array(0.0)
        ).reshape(3)
        start = mid + float(ap["start_m"]) * d
        start[1] = 0.0
        script = {"kind": "approach", "target": "fly", "speed": float(ap["speed_m_s"])}
    else:
        start, script = np.array([0.0, 0.0, -500.0]), None
    world.ents["person"] = Entity(
        id="person",
        kind="person",
        asset=ap["asset"],
        pos=start,
        yaw=0.0,
        radius=0.3,
        height=1.8,
        solid=False,
        script=script,
    )
    return world, scene


def render_all(v1: dict, out: Path) -> dict[str, np.ndarray]:
    dt = float(v1["frame_ms"]) / 1000.0
    n = int(v1["face_px"])
    frames: dict[str, np.ndarray] = {}
    world0, scene0 = build_world(v1, CONDS[0], 0)
    r = Renderer(run_dir=out / "render")
    try:
        ready = r.start()
        print(f"[v1] 렌더러 {ready}", flush=True)
        r.load_scene(scene0, world0.poses())
        for cond in CONDS:
            world, _ = build_world(v1, cond, 0)
            steps = int(
                round(
                    float(v1["rotation" if cond.startswith("rot") else "approach"]["window_ms"])
                    / float(v1["frame_ms"])
                )
            )
            turn = float(v1["rotation"]["turn_deg_s"])
            cmd = {"rot-left": Command(turn=turn), "rot-right": Command(turn=-turn)}.get(cond)
            seq = []
            t0 = time.time()
            for k in range(steps):
                world.step(dt, {"fly": cmd} if cmd else None)
                img, _ = r.render(k, world.poses(), world.eye_rig("fly"), n)
                seq.append(img)
            frames[cond] = np.stack(seq)
            print(f"[v1] 그림 {cond} · {steps} 걸음 · {time.time() - t0:.1f} s", flush=True)
    finally:
        r.stop()
    np.savez_compressed(out / "frames.npz", **{c.replace("-", "_"): f for c, f in frames.items()})
    return frames


def load_frames(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path / "frames.npz")
    return {c: z[c.replace("-", "_")] for c in CONDS}


def group_perm(kind: np.ndarray, sub: np.ndarray, side: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.arange(len(kind))
    for key in sorted(set(zip(kind, sub, side))):
        m = np.flatnonzero((kind == key[0]) & (sub == key[1]) & (side == key[2]))
        p[m] = rng.permutation(m)
    return p


def judge(rows: list[dict], v1: dict) -> dict:
    ps = v1["pass"]

    def vals(arm: str, cond: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["arm"] == arm and r["cond"] == cond], float)

    dl = vals("real", "rot-left", "DNa02_R") - vals("real", "rot-left", "DNa02_L")
    dr = vals("real", "rot-right", "DNa02_R") - vals("real", "rot-right", "DNa02_L")
    flips = int(np.sum(np.sign(dl) * np.sign(dr) < 0))
    real_abs = float(np.mean(np.abs(np.concatenate([dl, dr])))) if len(dl) else 0.0
    arms = sorted({r["arm"] for r in rows})
    rew = [a for a in arms if a.startswith("rewired")]
    rew_abs = {
        a: float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            vals(a, c, "DNa02_R") - vals(a, c, "DNa02_L")
                            for c in ("rot-left", "rot-right")
                        ]
                    )
                )
            )
        )
        for a in rew
    }
    r1 = flips >= int(ps["r1_flip_min"]) and all(
        v == 0 or v < float(ps["r1_rewired_max_ratio"]) * real_abs for v in rew_abs.values()
    )

    def lc4(arm: str) -> float:
        return float(
            sum(
                np.mean(vals(arm, f"app-{w}", "LC4_L") + vals(arm, f"app-{w}", "LC4_R"))
                for w in ("left", "right")
            )
        )

    real = lc4("real")
    per = {
        w: float(np.mean(vals("real", f"app-{w}", "LC4_L") + vals("real", f"app-{w}", "LC4_R")))
        for w in ("left", "right")
    }
    insufficient = any(v < float(ps["a1_insufficient"]) for v in per.values())
    shuf = {a: lc4(a) for a in arms if a.startswith("shuf")}
    med = float(np.median(list(shuf.values()))) if shuf else float("nan")
    a1 = (
        not insufficient
        and bool(shuf)
        and all(real > v for v in shuf.values())
        and (real / med if med > 0 else float("inf")) >= float(ps["a1_shuffle_median_ratio"])
    )
    rew_l = {a: lc4(a) for a in rew}
    a2 = bool(rew_l) and all(v < real for v in rew_l.values())
    lft = vals("real", "app-left", "LC4_L") - vals("real", "app-left", "LC4_R")
    rgt = vals("real", "app-right", "LC4_R") - vals("real", "app-right", "LC4_L")
    ipsi = int(np.sum((lft > 0) & (rgt > 0)))
    a3 = ipsi >= int(ps["a3_ipsi_min"])
    rep = {
        k: {c: float(np.mean(vals("real", c, f"{k}_L") + vals("real", c, f"{k}_R"))) for c in CONDS}
        for k in ("LPLC2", "GF", "DNp02", "HS", "VS")
    }
    return {
        "rule": v1["rule"],
        "R1": {
            "pass": bool(r1),
            "flips": flips,
            "n": len(dl),
            "left_RminusL": dl.tolist(),
            "right_RminusL": dr.tolist(),
            "real_mean_abs": real_abs,
            "rewired_mean_abs": rew_abs,
        },
        "A": {
            "pass": bool(a1 and a2 and a3),
            "A1": bool(a1),
            "A2": bool(a2),
            "A3": bool(a3),
            "insufficient": insufficient,
            "real_lc4": real,
            "per_stim": per,
            "shuffles": shuf,
            "shuffle_median": med,
            "rewired": rew_l,
            "ipsi_seeds": ipsi,
            "left_ipsi_minus_contra": lft.tolist(),
            "right_ipsi_minus_contra": rgt.tolist(),
        },
        "report": rep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=None, help="저장된 그림 폴더(다시 쓰기)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--maps", type=int, default=None)
    ap.add_argument("--brains", default=None)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    v1 = config.load("eyeworld")["v1"]
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [int(x) for x in v1["seeds"]]
    n_maps = int(v1["arms"]["shuffled_maps"]["n"]) if a.maps is None else a.maps
    brains = a.brains.split(",") if a.brains else ["real", *v1["arms"]["rewired"]]
    out = paths.runs_dir() / f"v1-check-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    dt_ms = float(v1["frame_ms"])
    rows: list[dict] = []
    with heavy("v1-check", extra_gb=5.0) as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        t0 = time.time()
        frames = load_frames(Path(a.frames)) if a.frames else render_all(v1, out)
        render_s = time.time() - t0
        base = graph_mod.load()
        lif = LIFParams.from_config()
        sched, fe = {}, None
        t0 = time.time()
        for cond in CONDS:
            fe = FrontendV2(base, int(v1["face_px"]), dt_ms)
            sched[cond] = np.stack([fe.step(f) for f in frames[cond]])
        fe_s = time.time() - t0
        assert fe is not None
        kind = np.concatenate([fe.t45.kind, fe.lob.kind])
        sub = np.concatenate([fe.t45.sub, fe.lob.sub])
        side = np.concatenate([fe.t45.side, fe.lob.side])
        s0 = int(v1["arms"]["shuffled_maps"]["seed0"])
        maps = [("real", np.arange(len(kind)))] + [
            (f"shuf{k}", group_perm(kind, sub, side, s0 + k)) for k in range(n_maps)
        ]
        typ = base.columns["type"].astype(str)
        sd = side_of(base)
        groups = {
            f"{k}_{s}": np.flatnonzero(np.isin(typ, v) & (sd == s))
            for k, v in READ.items()
            for s in ("L", "R")
        }
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "rule": v1["rule"],
                    "seeds": seeds,
                    "maps": [m for m, _ in maps],
                    "brains": brains,
                    "n_entry": int(len(fe.entry)),
                    "frontend": config.load("eyeworld")["frontend"]["version"],
                    "frames_sha16": {
                        c: hashlib.sha256(frames[c].tobytes()).hexdigest()[:16] for c in CONDS
                    },
                    "render_s": round(render_s, 1),
                    "frontend_s": round(fe_s, 1),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        brain_s = 0.0
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for brain in brains:
                g = graph_mod.brain_graph(base, brain)
                for seed in seeds:
                    eng = Engine(g, lif, device="cpu", seed=seed)
                    eng.set_poisson(fe.entry, np.zeros(len(fe.entry)))
                    for m, perm in maps if brain == "real" else maps[:1]:
                        for cond in CONDS:
                            t0 = time.time()
                            eng.reset()
                            for r in sched[cond]:
                                eng.set_rates(r[perm])
                                eng.run(dt_ms)
                            brain_s += time.time() - t0
                            _, neurons = eng.spikes()
                            cnt = np.bincount(neurons, minlength=g.n)
                            row = {
                                "arm": m if brain == "real" else brain,
                                "cond": cond,
                                "seed": seed,
                                **{k: int(cnt[ix].sum()) for k, ix in groups.items()},
                            }
                            rows.append(row)
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()
                            print(
                                f"[v1] {row['arm']:10} {cond:9} 씨앗 {seed}"
                                f" · DNa02 {row['DNa02_L']}/{row['DNa02_R']}"
                                f" · LC4 {row['LC4_L']}/{row['LC4_R']}"
                                f" · LPLC2 {row['LPLC2_L']}/{row['LPLC2_R']}"
                                f" · GF {row['GF_L']}/{row['GF_R']}"
                                f" · HS {row['HS_L']}/{row['HS_R']}",
                                flush=True,
                            )
    cost = {
        "render_s": round(render_s, 1),
        "frontend_s": round(fe_s, 1),
        "brain_s": round(brain_s, 1),
    }
    if not a.no_judge:
        res = judge(rows, v1)
        res["cost"] = cost
        (out / "judge.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[v1] 판정 R1 {res['R1']['pass']} · A {res['A']['pass']}"
            f"(A1 {res['A']['A1']} · A2 {res['A']['A2']} · A3 {res['A']['A3']})",
            flush=True,
        )
    print(f"[v1] 비용 {cost} · {out}")


if __name__ == "__main__":
    main()

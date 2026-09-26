"""그래프 분석 (R-7469 F-003) — 감각 입구에서 좌우 정보가 어디까지 남나.

`inspect laterality`: 감각 뉴런 종류에서 출발해 단계마다(감각 → 1차 → 2차) 시냅스가
**같은 쪽 / 반대쪽 / 가운데** 뉴런으로 가는 비율을 센다. 좌우 차가 회로에서 사라지는 자리를 찾는다.
좌우는 `loop.signals.side_of`(somaSide → rootSide → instance 끝)와 같은 규칙.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from . import graph as graph_mod
from . import paths


def _out_edges(g: graph_mod.Graph, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """출발 뉴런들의 나가는 연결 (출발, 도착, 시냅스 수)."""
    src, dst, cnt = [], [], []
    for i in idx:
        a, b = g.row_ptr[i], g.row_ptr[i + 1]
        src.append(np.full(b - a, i))
        dst.append(g.dst[a:b])
        cnt.append(g.count[a:b])
    if not src:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(cnt)


def laterality(g: graph_mod.Graph, types: list[str], steps: int = 2,
               top: int = 8, min_share: float = 0.02) -> dict[str, Any]:
    """감각 종류에서 `steps` 단계까지 — 단계마다 같은 쪽 · 반대쪽 시냅스 비율과 주요 도착 종류.

    다음 단계의 출발 뉴런 = 이번 단계 시냅스의 `min_share` 이상을 받은 종류(가운데 · 좌우 모두).
    """
    from .loop.signals import side_of

    side = side_of(g)
    ty = g.columns["type"]
    klass = g.columns["class"]
    start = g.select(type=types)
    out: dict[str, Any] = {"types": types, "start_neurons": int(len(start)),
                           "start_sides": dict(Counter(side[start].tolist())), "steps": []}
    frontier = start
    for step in range(1, steps + 1):
        s, d, c = _out_edges(g, frontier)
        s_side, d_side = side[s], side[d]
        lr = np.isin(s_side, ["L", "R"])
        same = lr & (s_side == d_side)
        opp = lr & np.isin(d_side, ["L", "R"]) & (s_side != d_side)
        mid = lr & ~same & ~opp
        total = float(c[lr].sum()) or 1.0
        by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"same": 0.0, "opp": 0.0,
                                                                    "mid": 0.0})
        for mask, key in ((same, "same"), (opp, "opp"), (mid, "mid")):
            for t, n in zip(ty[d[mask]], c[mask]):
                by_type[t or "-"][key] += float(n)
        ranked = sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))
        rows = []
        for t, v in ranked[:top]:
            tot = sum(v.values())
            cls = Counter(klass[d[ty[d] == t]].tolist()).most_common(1)[0][0] if t != "-" else ""
            rows.append({"type": t, "class": cls, "share": round(tot / total, 4),
                         "same": round(v["same"] / tot, 3), "opp": round(v["opp"] / tot, 3),
                         "mid": round(v["mid"] / tot, 3)})
        out["steps"].append({
            "step": step, "from_neurons": int(len(frontier)), "synapses": int(total),
            "same": round(float(c[same].sum()) / total, 4),
            "opp": round(float(c[opp].sum()) / total, 4),
            "mid": round(float(c[mid].sum()) / total, 4),
            "top_targets": rows,
        })
        keep = [t for t, v in ranked if sum(v.values()) / total >= min_share and t != "-"]
        frontier = g.select(type=keep) if keep else np.zeros(0, int)
        out["steps"][-1]["next_types"] = keep
    return out


ODOR_CONTROL_RULE: dict[str, Any] = {
    "version": "odor-control-v1",
    "stim_types": ["ORN_DM1", "ORN_VA2"],   # 먹이(식초) 냄새 ORN — task.yaml smell 과 같다
    "readouts": {                                     # 좌우를 읽을 자리(연구 노트 14 §5 · M1)
        "PN": {"class": ["ALPN"]},
        "LN4": {"type": ["v2LN3A1_b", "lLN2X02", "lLN15", "lLN2F_b"]},
        "DNa02": {"type": ["DNa02"]},
    },
    "pass_accuracy": 0.9,   # 진짜 배선 부호 정답률이 이 이상 + 재배선보다 높으면
}


def odor_control(graphs: dict[str, graph_mod.Graph], rates: list[float], reps: int,
                 window_ms: float, rule: dict[str, Any] = ODOR_CONTROL_RULE,
                 on_trial=None) -> dict[str, Any]:
    """양성 대조(M1) — ORN 을 **한쪽만** 구동하고 읽는 자리의 좌우 발화 차 부호를 센다.

    한쪽 100 % 자극도 좌우를 못 나르면(정답률 0.5) 냄새 쪽 문제는 입력 기하 이전(회로 · 모델)이다.
    뇌마다 시행 = 쪽 2 × 발화율 × 반복. 반복마다 포아송 씨앗이 다르다.
    """
    from .loop.signals import side_of
    from .sim import Engine, LIFParams

    params = LIFParams.from_config()
    out: dict[str, Any] = {"rule": rule, "rates_hz": rates, "reps": reps,
                           "window_ms": window_ms, "sim": params.to_dict(), "brains": {}}
    for name, g in graphs.items():
        side = side_of(g)
        stim = g.select(type=rule["stim_types"])
        groups = {k: g.select(**sel) for k, sel in rule["readouts"].items()}
        eng = Engine(g, params, seed=0)
        trials = []
        for rate in rates:
            for stim_side in ("L", "R"):
                idx = stim[side[stim] == stim_side]
                for rep in range(reps):
                    eng.set_poisson(idx, rate)
                    eng.rng = np.random.default_rng(10_000 * rep + int(rate))
                    eng.reset()
                    t0 = time.time()
                    eng.run(window_ms)
                    _, neurons = eng.spikes()
                    counts = np.bincount(neurons, minlength=g.n)
                    row: dict[str, Any] = {"rate_hz": rate, "side": stim_side, "rep": rep,
                                           "wall_s": round(time.time() - t0, 2)}
                    for k, idx_g in groups.items():
                        hz = {s: float(counts[idx_g[side[idx_g] == s]].mean()
                                       / (window_ms / 1000.0))
                              if np.any(side[idx_g] == s) else 0.0 for s in ("L", "R")}
                        other = "R" if stim_side == "L" else "L"
                        diff = hz[stim_side] - hz[other]
                        row[k] = {"L_hz": round(hz["L"], 3), "R_hz": round(hz["R"], 3),
                                  "same_minus_other_hz": round(diff, 3),
                                  "correct": 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)}
                    trials.append(row)
                    if on_trial is not None:
                        on_trial(name, row)
        summary = {}
        for k in groups:
            by_rate = {}
            for rate in rates:
                rs = [t[k] for t in trials if t["rate_hz"] == rate]
                by_rate[str(rate)] = {
                    "accuracy": round(float(np.mean([r["correct"] for r in rs])), 3),
                    "diff_hz_mean": round(float(np.mean(
                        [r["same_minus_other_hz"] for r in rs])), 3),
                    "rate_hz_mean": round(float(np.mean(
                        [(r["L_hz"] + r["R_hz"]) / 2 for r in rs])), 3)}
            summary[k] = {"neurons": {s: int(np.sum(side[groups[k]] == s)) for s in ("L", "R")},
                          "accuracy": round(float(np.mean([t[k]["correct"] for t in trials])), 3),
                          "by_rate": by_rate}
        out["brains"][name] = {"stim_neurons": {s: int(np.sum(side[stim] == s)) for s in "LR"},
                               "summary": summary, "trials": trials}
    real = out["brains"].get("real", {}).get("summary", {})
    rew = [b["summary"] for n, b in out["brains"].items() if n != "real"]
    out["verdict"] = {k: {"real": real[k]["accuracy"],
                          "rewired": [r[k]["accuracy"] for r in rew],
                          "carries_side": bool(real[k]["accuracy"] >= rule["pass_accuracy"]
                                               and all(real[k]["accuracy"] > r[k]["accuracy"]
                                                       for r in rew))}
                      for k in real}
    return out


def main_odor_control(args: argparse.Namespace) -> int:
    from .guard import GuardRefused, heavy

    rates = [float(r) for r in args.rates.split(",")]
    try:
        with heavy("odor-control") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load(args.graph)
            graphs = {"real": g}
            for s in [int(x) for x in args.rewired_seeds.split(",") if x != ""]:
                graphs[f"rewired-s{s}"] = graph_mod.rewire(g, s)
            t0 = time.time()

            def say(name: str, row: dict[str, Any]) -> None:
                parts = " · ".join(f"{k} {row[k]['L_hz']:.1f}/{row[k]['R_hz']:.1f}"
                                   for k in ODOR_CONTROL_RULE["readouts"])
                print(f"[{name}] {row['rate_hz']:.0f} Hz · 자극 {row['side']} · "
                      f"반복 {row['rep']} · "
                      f"{parts}(L/R Hz) · {row['wall_s']} s", flush=True)

            res = odor_control(graphs, rates, args.reps, args.window_ms, on_trial=say)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    res.update({"graph": args.graph, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_s": round(time.time() - t0, 1)})
    path = paths.runs_dir() / f"odor-control-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res["verdict"].items():
        print(f"{k:<6} 진짜 {v['real']:.3f} · 재배선 {v['rewired']} · "
              f"{'좌우를 나른다' if v['carries_side'] else '못 나름'}")
    print(f"기록 {path} · {res['wall_s']} s")
    return 0


def main(args: argparse.Namespace) -> int:
    g = graph_mod.load(args.graph)
    res = laterality(g, [t for t in args.types.split(",") if t], steps=args.steps)
    res.update({"graph": g.name, "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    path = paths.runs_dir() / f"inspect-laterality-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"시작 {res['types']} · 뉴런 {res['start_neurons']} · 좌우 {res['start_sides']}")
    for st in res["steps"]:
        print(f"[단계 {st['step']}] 출발 {st['from_neurons']} · 시냅스 {st['synapses']:,} · "
              f"같은 쪽 {st['same']:.3f} · 반대쪽 {st['opp']:.3f} · 가운데 {st['mid']:.3f}")
        for r in st["top_targets"]:
            print(f"    {r['type']:<22}{r['class']:<18}몫 {r['share']:.3f} · "
                  f"같은 {r['same']:.2f} · 반대 {r['opp']:.2f} · 가운데 {r['mid']:.2f}")
    print(f"기록 {path}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("inspect", help="그래프 분석(가벼움 · CPU)")
    isub = p.add_subparsers(dest="inspect_cmd", required=True)
    lat = isub.add_parser("laterality",
                          help="감각 입구에서 좌우 정보가 어디까지 남나(R-7469 F-003)")
    lat.add_argument("--types", default="ORN_DM1,ORN_VA2")
    lat.add_argument("--steps", type=int, default=2)
    lat.add_argument("--graph", default="traced-v1")
    lat.set_defaults(func=main)
    oc = isub.add_parser("odor-control",
                         help="양성 대조 — ORN 한쪽만 구동해 좌우가 남나(노트 14 M1 · 무거움)")
    oc.add_argument("--rates", default="16,40,150", help="자극 발화율(Hz) 목록")
    oc.add_argument("--reps", type=int, default=10)
    oc.add_argument("--window-ms", type=float, default=1000.0)
    oc.add_argument("--rewired-seeds", default="0")
    oc.add_argument("--graph", default="traced-v1")
    oc.set_defaults(func=main_odor_control)
    from .loop import bump

    bump.register(isub)

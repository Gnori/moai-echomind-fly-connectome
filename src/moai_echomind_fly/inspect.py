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


ODOR_CONTRAST_RULE: dict[str, Any] = {
    "version": "odor-contrast-v1",
    "stim_types": ODOR_CONTROL_RULE["stim_types"],
    "readouts": {"ORN": {"type": ODOR_CONTROL_RULE["stim_types"]},   # 입력 자체(상한 비교용)
                 "PN": {"class": ["ALPN"]},
                 "LN4": {"type": ["v2LN3A1_b", "lLN2X02", "lLN15", "lLN2F_b"]}},
    "base_hz": 40.0,
    # 판정: 한 개 빼고 문턱(두 쪽 평균의 가운데) 정답률 — M1 에서 LN4 가 바탕이 오른쪽으로 치우쳐
    # "자극한 쪽이 더 높나" 부호 규칙이 맞지 않았다(사후)
    # → 이 판부터는 결과 보기 전에 이 방식으로 정한다
    "pass_accuracy": 0.75,
}


def _loo_threshold_acc(a: np.ndarray, b: np.ndarray) -> float:
    """a(왼쪽 강함 시행의 L−R) 대 b(오른쪽 강함) — 한 개 빼고 가운데 문턱으로 맞힌 비율."""
    hit = 0.0
    for i in range(len(a)):
        th = (np.delete(a, i).mean() + b.mean()) / 2
        hit += 1.0 if a[i] > th else (0.5 if a[i] == th else 0.0)
    for i in range(len(b)):
        th = (a.mean() + np.delete(b, i).mean()) / 2
        hit += 1.0 if b[i] < th else (0.5 if b[i] == th else 0.0)
    return hit / (len(a) + len(b))


def odor_contrast(g: graph_mod.Graph, contrasts: list[float], windows_ms: list[float],
                  reps: int, rule: dict[str, Any] = ODOR_CONTRAST_RULE,
                  on_trial=None) -> dict[str, Any]:
    """약한 대조(M1b) — 양쪽 ORN 을 공통 발화율로 구동하되 좌우를 `c` 만큼 다르게.

    왼쪽 강함: L = base·(1 + c/2) · R = base·(1 − c/2), 오른쪽 강함은 반대.
    창마다 좌우 평균 발화 차를 읽고, 두 쪽 시행이 갈리는지(한 개 빼고 문턱 · d′) 본다.
    실제 과제의 좌우 차는 1~5 %(노트 14 §2 · §3.1).
    """
    from .loop.signals import side_of
    from .sim import Engine, LIFParams

    params = LIFParams.from_config()
    side = side_of(g)
    stim = g.select(type=rule["stim_types"])
    groups = {k: g.select(**sel) for k, sel in rule["readouts"].items()}
    eng = Engine(g, params, seed=0)
    base = rule["base_hz"]
    cells: dict[str, Any] = {}
    for c in contrasts:
        for w in windows_ms:
            diffs: dict[str, dict[str, list[float]]] = {k: {"L": [], "R": []} for k in groups}
            for strong in ("L", "R"):
                hi, lo = base * (1 + c / 2), base * (1 - c / 2)
                rates = np.where(side[stim] == strong, hi, lo)
                for rep in range(reps):
                    eng.set_poisson(stim, rates)
                    eng.rng = np.random.default_rng(100_000 * rep + int(1000 * c) + int(w))
                    eng.reset()
                    t0 = time.time()
                    eng.run(w)
                    _, neurons = eng.spikes()
                    counts = np.bincount(neurons, minlength=g.n)
                    for k, ix in groups.items():
                        hz = [counts[ix[side[ix] == s]].mean() / (w / 1000.0) for s in "LR"]
                        diffs[k][strong].append(float(hz[0] - hz[1]))
                    if on_trial is not None:
                        on_trial(c, w, strong, rep, round(time.time() - t0, 2))
            cell = {}
            for k, d in diffs.items():
                a, b = np.array(d["L"]), np.array(d["R"])
                sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) if reps > 1 else 0.0
                cell[k] = {"acc": round(_loo_threshold_acc(a, b), 3),
                           "d_prime": round(float((a.mean() - b.mean()) / sd), 2) if sd else None,
                           "lr_diff_hz_mean": {"L_strong": round(float(a.mean()), 3),
                                               "R_strong": round(float(b.mean()), 3)},
                           "sd_hz": round(sd, 3)}
            cells[f"c{c:g}|w{w:g}"] = {"contrast": c, "window_ms": w, **cell}
    return {"rule": rule, "contrasts": contrasts, "windows_ms": windows_ms, "reps": reps,
            "sim": params.to_dict(),
            "stim_neurons": {s: int(np.sum(side[stim] == s)) for s in "LR"}, "cells": cells}


def main_odor_contrast(args: argparse.Namespace) -> int:
    from .guard import GuardRefused, heavy

    contrasts = [float(x) for x in args.contrasts.split(",")]
    windows = [float(x) for x in args.windows_ms.split(",")]
    try:
        with heavy("odor-contrast") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load(args.graph)
            t0 = time.time()

            def say(c, w, strong, rep, sec):
                if rep == 0:
                    print(f"[contrast {c:g} · 창 {w:g} ms · {strong} 강함] 첫 시행 {sec} s",
                          flush=True)

            res = odor_contrast(g, contrasts, windows, args.reps, on_trial=say)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    res.update({"graph": args.graph, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_s": round(time.time() - t0, 1)})
    path = paths.runs_dir() / f"odor-contrast-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    ks = list(ODOR_CONTRAST_RULE["readouts"])
    print("좌우 차 · 창     " + "".join(f"{k:>16}" for k in ks) + "   (정답률 / d′)")
    for key, c in res["cells"].items():
        print(f"{key:<16}" + "".join(f"{c[k]['acc']:>9.3f} / {c[k]['d_prime'] or 0:>4.1f}"
                                      for k in ks))
    print(f"기록 {path} · {res['wall_s']} s")
    return 0


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
    ct = isub.add_parser("odor-contrast",
                         help="약한 좌우 차에서 좌우가 남나(노트 14 M1b · 무거움)")
    ct.add_argument("--contrasts", default="0.02,0.05,0.1,0.2")
    ct.add_argument("--windows-ms", default="300,1000,3000")
    ct.add_argument("--reps", type=int, default=10)
    ct.add_argument("--graph", default="traced-v1")
    ct.set_defaults(func=main_odor_contrast)
    from .loop import bump

    bump.register(isub)

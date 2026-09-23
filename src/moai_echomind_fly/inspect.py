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
    from .loop import bump

    bump.register(isub)

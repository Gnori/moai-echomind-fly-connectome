"""시야 자리 대응 retino-v1 (R-7469 F-008 2단계) — 시각 투사 뉴런마다 고도를 연결 지도로 계산한다.

규칙 · 근거 · 한계는 `config/retino.yaml`. 결과는 `data/graph/<그래프>.<판>.json`(I-7469 §3.3).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.feather as feather

from . import config, data, paths
from . import graph as graph_mod


def column_hex(g: graph_mod.Graph, raw: Path | None = None) -> np.ndarray:
    """그래프 인덱스마다 공식 기둥 좌표 (hex1, hex2) — 없으면 NaN."""
    raw = raw or paths.raw_dir()
    name = {s.key: s.name for s in data.sources()}["annotations"]
    t = feather.read_table(raw / name, columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    bid = t.column("bodyId").to_numpy()
    h = np.stack([np.array([np.nan if v is None else v for v in t.column(c).to_pylist()], float)
                  for c in ("assignedOlHex1", "assignedOlHex2")], axis=1)
    pos = np.searchsorted(g.body_ids, bid)
    ok = pos < g.n
    ok[ok] &= g.body_ids[pos[ok]] == bid[ok]
    out = np.full((g.n, 2), np.nan)
    out[pos[ok]] = h[ok]
    return out


def weighted_input_hex(g: graph_mod.Graph, targets: np.ndarray, known: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """뉴런마다 좌표를 가진 입력의 시냅스 가중 평균 좌표 · 그 입력 시냅스 수."""
    src = g.src()
    m = np.isin(g.dst, targets) & ~np.isnan(known[src, 0])
    s, d, c = src[m], g.dst[m], g.count[m].astype(np.float64)
    order = {int(t): k for k, t in enumerate(targets)}
    k = np.array([order[int(x)] for x in d], dtype=np.int64)
    w = np.bincount(k, weights=c, minlength=len(targets))
    hx = np.stack([np.bincount(k, weights=c * known[s, j], minlength=len(targets))
                   for j in (0, 1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return hx / w[:, None], w


def build(g: graph_mod.Graph, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    from .loop.signals import side_of

    cfg = cfg or config.load("retino")
    hexes = column_hex(g)
    known = hexes.copy()
    ty = g.columns["type"].astype(str)
    two = np.flatnonzero(np.logical_or.reduce(
        [np.char.startswith(ty, p) for p in cfg["two_hop_types_prefix"]]))
    hx2, _ = weighted_input_hex(g, two, hexes)
    known[two] = hx2
    mask = np.ones(g.n, bool)
    for col, vals in cfg["select"].items():
        mask &= np.isin(g.columns[col], vals)
    targets = np.flatnonzero(mask)
    hx, w = weighted_input_hex(g, targets, known)
    cols = hexes[~np.isnan(hexes[:, 0])]
    ys = cols.sum(1)
    y_min, y_max = float(ys.min()), float(ys.max())
    lo, hi = cfg["elevation_range_deg"]
    side = side_of(g)
    neurons, missing = [], Counter()
    for i, t in enumerate(targets):
        if w[i] < cfg["min_input_synapses"] or np.isnan(hx[i, 0]):
            missing[str(g.columns["type"][t])] += 1
            continue
        y = float(hx[i].sum())
        neurons.append({"idx": int(t), "bodyId": int(g.body_ids[t]),
                        "type": str(g.columns["type"][t]), "side": str(side[t]),
                        "hex": [round(float(hx[i, 0]), 3), round(float(hx[i, 1]), 3)],
                        "elevation_deg": round(lo + (y - y_min) / (y_max - y_min) * (hi - lo), 2),
                        "input_synapses": int(w[i])})
    return {"rule": cfg["version"], "graph": g.name, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "assignedOlHex1/2(공식 주석) · 입력 시냅스 가중 평균 · T4/T5 는 두 단계",
            "y_range": [y_min, y_max], "elevation_range_deg": [lo, hi],
            "azimuth": "확정 못 함 — 눈 단위(좌/우)만(config/retino.yaml)",
            "neurons": neurons, "missing": dict(missing)}


def path_for(graph_name: str, rule: str) -> Path:
    return paths.graph_dir() / f"{graph_name}.{rule}.json"


def load(graph_name: str = "traced-v1", rule: str | None = None) -> dict[str, Any]:
    ver = rule or str(config.load("retino")["version"])
    return json.loads(path_for(graph_name, ver).read_text(encoding="utf-8"))


def main(args: argparse.Namespace) -> int:
    g = graph_mod.load(args.graph)
    res = build(g)
    out = path_for(g.name, res["rule"])
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    by_type: dict[str, list[float]] = {}
    for n in res["neurons"]:
        by_type.setdefault(n["type"], []).append(n["elevation_deg"])
    n_miss = sum(res["missing"].values())
    print(f"[retino] {res['rule']} · 뉴런 {len(res['neurons'])} · 빠짐 {n_miss}"
          f" · 기둥 y {res['y_range']} → 고도 {res['elevation_range_deg']}")
    for t in ("LC4", "LC6", "LC11", "LC12", "LC16", "LC17", "LPLC1", "LPLC2"):
        v = by_type.get(t, [])
        if v:
            print(f"  {t:<7} {len(v):>4} · 고도 중앙 {np.median(v):6.1f}° · "
                  f"10~90 % {np.percentile(v, 10):6.1f} ~ {np.percentile(v, 90):6.1f}°")
    print(f"[retino] 저장 {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("retinotopy", help="시야 자리 대응(R-7469 F-008) — 가벼움 · CPU")
    rs = p.add_subparsers(dest="retino_cmd", required=True)
    b = rs.add_parser("build", help="시각 투사 뉴런마다 고도 → data/graph/<그래프>.<판>.json")
    b.add_argument("--graph", default="traced-v1")
    b.set_defaults(func=main)

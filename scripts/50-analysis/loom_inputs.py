"""다가옴 길 입력 구성(10-16 · 조회만 · 판정 아님) — V0b 설계 재료.

LPLC2 · LC4 · GF(DNp01) 마다 입력 시냅스를 보낸 종류별로 · 그 종류에 공식 기둥 좌표가 있는 비율 ·
좌표 있는 입력 가중 평균으로 자리를 얻는 비율(T4/T5 와 같은 식) · 상위 종류의 상위 종류(두 단계)도.
사용: python scripts/50-analysis/loom_inputs.py → runs/loom-inputs-<시각>.json
"""

from __future__ import annotations

import collections
import json
import time

import numpy as np

from moai_echomind_fly import config, paths, retino
from moai_echomind_fly import graph as graph_mod

TARGETS = {"LPLC2": ["LPLC2"], "LC4": ["LC4"], "GF": ["DNp01"]}


def main() -> None:
    g = graph_mod.load()
    typ = g.columns["type"].astype(str)
    sc = g.columns["superclass"].astype(str)
    src = g.src()
    hexes = retino.column_hex(g)
    has = ~np.isnan(hexes[:, 0])
    min_syn = float(config.load("retino")["min_input_synapses"])
    out: dict = {}
    for name, types in TARGETS.items():
        tgt = np.flatnonzero(np.isin(typ, types))
        m = np.isin(g.dst, tgt)
        c = collections.Counter()
        for t, n in zip(typ[src[m]], g.count[m]):
            c[str(t)] += int(n)
        total = sum(c.values())
        rows = []
        for t, n in c.most_common(15):
            idx = np.flatnonzero(typ == t)
            _, w = retino.weighted_input_hex(g, idx, hexes)
            rows.append({"type": t, "share": round(n / total, 4), "neurons": int(len(idx)),
                         "superclass": str(sc[idx[0]]) if len(idx) else "",
                         "official_coord": round(float(has[idx].mean()), 3) if len(idx) else 0.0,
                         "weighted_coord": round(float((w >= min_syn).mean()), 3) if len(idx) else 0.0})
        out[name] = {"total_syn": total, "top": rows}
        print(f"[loom] {name} 입력 {total:,}", flush=True)
        for r in rows:
            print(f"   {r['type']:12} {r['share']:6.1%} · 뉴런 {r['neurons']:5} · {r['superclass']:18} · "
                  f"공식 좌표 {r['official_coord']:.0%} · 가중 좌표 {r['weighted_coord']:.0%}", flush=True)
    dst = paths.runs_dir() / f"loom-inputs-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[loom] {dst}")


if __name__ == "__main__":
    main()

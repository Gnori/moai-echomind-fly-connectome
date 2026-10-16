"""V0 전 연결 지도 확인(10-16 · 조회만 · 판정 아님) — 연구 노트 17 §8.4 ①~④.

① T4a–d · T5a–d 기둥 자리: 공식 좌표 없음 → 좌표 있는 입력 기둥의 시냅스 가중 평균(retino 모듈과 같은 식) · 얻는 비율.
② VS · HS 입력 구성(종류별 몫) — 10-16 재기에서 T4a 로 VS 가 켜진 길.
③ HS · VS → DNa02 직접 연결 · 두 단계 길(중간 종류).
④ LPLC2 · LC4 입력 가운데 T4/T5 몫(MaleCNS).
사용: python scripts/50-analysis/v0_graph_checks.py → runs/v0-graph-checks-<시각>.json
"""

from __future__ import annotations

import collections
import json
import time

import numpy as np

from moai_echomind_fly import config, paths, retino
from moai_echomind_fly import graph as graph_mod

T45 = [f"T{k}{d}" for k in (4, 5) for d in "abcd"]


def inputs_by_type(g, typ: np.ndarray, src: np.ndarray, targets: np.ndarray, top: int = 12) -> dict:
    m = np.isin(g.dst, targets)
    c = collections.Counter()
    for t, n in zip(typ[src[m]], g.count[m]):
        c[str(t)] += int(n)
    total = sum(c.values())
    t45 = sum(v for k, v in c.items() if k in T45)
    return {"total_syn": total, "t4t5_share": round(t45 / total, 4) if total else 0.0,
            "t4t5_by_subtype": {k: round(c[k] / total, 4) for k in T45 if c[k]},
            "top": [(k, v, round(v / total, 4)) for k, v in c.most_common(top)]}


def main() -> None:
    g = graph_mod.load()
    typ = g.columns["type"].astype(str)
    side = g.columns["somaSide"].astype(str)
    src = g.src()
    out: dict = {}
    min_syn = float(config.load("retino")["min_input_synapses"])
    known = retino.column_hex(g)
    cov = {}
    for t in T45:
        for s in ("L", "R"):
            idx = np.flatnonzero((typ == t) & (side == s))
            hx, w = retino.weighted_input_hex(g, idx, known)
            ok = (w >= min_syn) & ~np.isnan(hx[:, 0])
            cov[f"{t}-{s}"] = {"n": int(len(idx)), "with_coord": int(ok.sum()),
                              "median_input_syn": float(np.median(w)) if len(w) else 0.0}
    out["t4t5_coords"] = cov
    tot = sum(v["n"] for v in cov.values())
    got = sum(v["with_coord"] for v in cov.values())
    print(f"① T4/T5 좌표 얻음 {got}/{tot}({got / tot:.1%}) · 최소 입력 시냅스 {min_syn:.0f}", flush=True)
    for name, types in (("HS", ["HSE", "HSN", "HSS"]), ("VS", ["VS"]), ("LPLC2", ["LPLC2"]),
                        ("LC4", ["LC4"]), ("DNa02", ["DNa02"])):
        idx = np.flatnonzero(np.isin(typ, types))
        r = inputs_by_type(g, typ, src, idx)
        out[f"inputs_{name}"] = r
        print(f"②④ {name}: 입력 시냅스 {r['total_syn']:,} · T4/T5 몫 {r['t4t5_share']:.1%} · "
              f"아형 {r['t4t5_by_subtype']} · 상위 {[(k, p) for k, _, p in r['top'][:6]]}", flush=True)
    dna = np.flatnonzero(typ == "DNa02")
    for name, types in (("HS", ["HSE", "HSN", "HSS"]), ("VS", ["VS"])):
        a = np.isin(typ, types)
        direct = int(g.count[a[src] & np.isin(g.dst, dna)].sum())
        mid = collections.Counter()
        e1 = a[src] & (g.count >= 5)
        first = collections.Counter()
        for d_, n in zip(g.dst[e1], g.count[e1]):
            first[int(d_)] += int(n)
        into = np.isin(g.dst, dna) & (g.count >= 5)
        second = collections.Counter()
        for s_, n in zip(src[into], g.count[into]):
            second[int(s_)] += int(n)
        for k in set(first) & set(second):
            mid[str(typ[k])] += min(first[k], second[k])
        out[f"{name}_to_DNa02"] = {"direct_syn": direct, "two_hop_mid_types": mid.most_common(10)}
        print(f"③ {name} → DNa02 직접 {direct} 시냅스 · 두 단계 중간(≥5 · 최소 몫) {mid.most_common(6)}", flush=True)
    dst = paths.runs_dir() / f"v0-graph-checks-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v0-checks] {dst}")


if __name__ == "__main__":
    main()

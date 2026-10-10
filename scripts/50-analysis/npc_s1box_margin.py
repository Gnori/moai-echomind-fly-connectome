"""S1 상자 월드 — 에피소드마다 가장 가까운 거리 · 반사 지연을 줄이면 몇 번 피하나(어림 · 10-10 · 가능성 먼저).

월드 기록(world_exp.jsonl · echomind-world 상자 월드)의 steer 응답 좌표로 NPC 와 위험 몸 중심 거리를 결정 구간마다 잰다.
어림: 반사가 한 구간(Δt) 빨라지면 옆으로 더 가는 거리 = 비켜서기 속도 × Δt → 그만큼 가장 가까운 거리에 더한다
(위험이 곧게 오고 · 반사가 켜지는 시점은 그대로라는 가정 · 판정 아님).
사용: python scripts/50-analysis/npc_s1box_margin.py <world_exp.jsonl> [시작 HH:MM 날짜 YYYY-MM-DD] → runs/npc-s1box-margin-<시각>.json
"""

from __future__ import annotations

import collections
import datetime
import json
import math
import statistics as st
import sys
import time

from moai_echomind_fly import paths

LINE_M = 0.595          # npc-s1-v1 부딪힘 판정선(반경 0.35 + 0.225 + 0.02 · R-5138)
SIDESTEP_M_S = 4.5      # npc-v3 reflex_retreat 30 mm/s × 0.15 m/mm
GAINS_S = [0.0, 0.025, 0.05]   # 반사가 이만큼 빨라지면


def main(path: str, start: str = "2026-10-04 10:25:28", end: str = "2026-10-04 13:40") -> None:
    t0 = datetime.datetime.fromisoformat(start).timestamp()
    t1 = datetime.datetime.fromisoformat(end).timestamp()
    eps, cur = [], None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not t0 <= r.get("wall", 0) <= t1:
            continue
        a, g = r.get("ask") or {}, r.get("got") or {}
        if r["op"] == "reset":
            cur = {"seed": r["seed"], "npc": {}, "haz": {}, "label": None, "escape": 0}
            eps.append(cur)
        elif cur is None:
            continue
        elif r["op"] == "steer":
            who = "npc" if a.get("agent_id") == "npc" else "haz"
            cur[who][round(r["sim_s"], 2)] = (g["x"], g["z"])
        elif r["op"] == "neural":
            cur["label"] = a.get("label") or cur["label"]
            cur["escape"] += bool(a.get("escape"))
    rows = collections.defaultdict(list)
    for e in eps:
        ts = sorted(set(e["npc"]) & set(e["haz"]))
        if ts:
            rows[e["label"]].append({"seed": e["seed"], "escape_decisions": e["escape"],
                                     "min_m": round(min(math.dist(e["npc"][t], e["haz"][t])
                                                        for t in ts), 4)})
    out = {"source": path, "window": [start, end], "line_m": LINE_M, "sidestep_m_s": SIDESTEP_M_S,
           "configs": {}}
    for k, v in rows.items():
        mins = [x["min_m"] for x in v]
        est = {f"{g:g}s": sum(m + SIDESTEP_M_S * g >= LINE_M for m in mins) for g in GAINS_S}
        out["configs"][k] = {"episodes": len(v), "min_median_m": round(st.median(mins), 4),
                             "min_range_m": [min(mins), max(mins)],
                             "reflex_episodes": sum(x["escape_decisions"] > 0 for x in v),
                             "avoid_if_earlier": est, "rows": v}
        print(f"[margin] {k} {len(v)} · 가장 가까운 거리 중앙 {st.median(mins):.3f} m · "
              f"반사 {out['configs'][k]['reflex_episodes']} · 빨라지면 피함 {est}", flush=True)
    dst = paths.runs_dir() / f"npc-s1box-margin-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[margin] {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])

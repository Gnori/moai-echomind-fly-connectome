"""관문(F-004) 사후 재판정 — 선형 검사에 절편을 넣으면 통과 채널이 바뀌나(10-12 · 판정 바꾸지 않음).

S0b 촉각 착시(runs/touch-anomaly-*)와 같은 까닭이 1판 입력 채널 고르기에도 있었는지 본다.
원판 gate.json 은 덮어쓰지 않는다. 사용: uv run python scripts/50-analysis/gate_intercept.py fly-loop-v4
→ runs/gate-intercept-<자료>-<시각>.json
"""

from __future__ import annotations

import json
import sys
import time

from moai_echomind_fly import paths
from moai_echomind_fly.loop.compress import gate, out_dir


def main(dataset: str) -> None:
    old = json.loads((out_dir(dataset) / "gate.json").read_text(encoding="utf-8"))
    new = gate(dataset, intercept=True, write=False)
    rows = {}
    for k, c in new["cells"].items():
        o = old["cells"].get(k, {})
        rows[k] = {"real_old": o.get("real"), "real_new": c["real"],
                   "rewired_old": (o.get("rewired") or {}).get("values"),
                   "rewired_new": c["rewired"]["values"], "baseline": c["baseline"],
                   "pass_old": o.get("pass"), "pass_new": c["pass"], "p_holm_new": c.get("p_holm")}
    out = {"dataset": dataset, "passed_channels_old": old["passed_channels"],
           "passed_channels_new": new["passed_channels"],
           "passed_cells_old": old["passed_cells"], "passed_cells_new": new["passed_cells"],
           "cells": rows}
    print(f"[gate-intercept] 통과 채널 원판 {old['passed_channels']} → 절편 {new['passed_channels']}")
    print(f"[gate-intercept] 통과 칸 원판 {old['passed_cells']}")
    print(f"[gate-intercept] 통과 칸 절편 {new['passed_cells']}")
    for k, r in rows.items():
        if r["pass_old"] != r["pass_new"] or (r["real_old"] is not None and r["real_old"] < r["baseline"]):
            print(f"[gate-intercept] {k}: 진짜 {r['real_old']} → {r['real_new']:.3f} · 섞은 "
                  f"{r['rewired_old']} → {r['rewired_new']} · 찍기 {r['baseline']:.3f} · "
                  f"통과 {r['pass_old']} → {r['pass_new']}", flush=True)
    dst = paths.runs_dir() / f"gate-intercept-{dataset}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[gate-intercept] {dst}")


if __name__ == "__main__":
    main(sys.argv[1])

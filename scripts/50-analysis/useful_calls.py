"""층 구조 성적표의 결정 기록 → 호출이 쓸모 있었나(연구 노트 15 T3b · R-1027).

부른 결정마다: Qwen 행동이 빠른 정책 행동(`fast_action` · 같은 결정에서 계산해 둔 것)과 달랐나,
그리고 교사와 맞았나. 계기(trigger)별로 센다. 같은 호출률 무작위 구성과 나란히 본다.
사용: uv run python scripts/50-analysis/useful_calls.py <성적표 폴더>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def key(a: dict | None) -> tuple:
    a = a or {}
    return (a.get("turn", "none"), bool(a.get("stop", False)))


def summarize(path: Path) -> dict:
    by = defaultdict(lambda: {"calls": 0, "differs": 0, "llm_teacher": 0, "fast_teacher": 0})
    n = calls = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            if r.get("tier") != "llm":
                continue
            calls += 1
            b = by[r.get("trigger") or "?"]
            b["calls"] += 1
            b["differs"] += int(key(r.get("action")) != key(r.get("fast_action")))
            b["llm_teacher"] += int(key(r.get("action")) == key(r.get("teacher")))
            b["fast_teacher"] += int(key(r.get("fast_action")) == key(r.get("teacher")))
    return {"decisions": n, "calls": calls, "by_trigger": dict(by)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    d = Path(ap.parse_args().folder)
    files = sorted(d.glob("*.decisions.jsonl"))
    shown = 0
    for p in files:
        s = summarize(p)
        if not s["calls"] or not any(k != "always" for k in s["by_trigger"]):
            continue
        shown += 1
        print(f"{p.name.split('.')[0]:28} 결정 {s['decisions']} · 호출 {s['calls']} "
              f"({s['calls'] / max(s['decisions'], 1):.3f})")
        for t, b in sorted(s["by_trigger"].items()):
            c = max(b["calls"], 1)
            print(f"  {t:10} 호출 {b['calls']:4} · 빠른 정책과 다름 {b['differs'] / c:.3f} · "
                  f"교사와 같음 LLM {b['llm_teacher'] / c:.3f} 대 빠른 정책 {b['fast_teacher'] / c:.3f}")
    if not shown:
        print("층 구성 결정 기록이 없다(tier · trigger 칸) — scorecard-v4 층 구성에서만 의미가 있다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

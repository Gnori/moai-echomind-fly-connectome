"""성적표 결정 기록(scorecard-v3)으로 — 교사가 돌라 한 쪽과 Qwen 이 돈 쪽(가벼움 · 파일만).

냄새 맡기 구간(task.yaml episode.sniff_decisions · 몸을 세워 둔 결정) 뒤, 교사가 left/right 를 준 결정만 센다.
"같은 쪽"이 0.5 근처면 도는 방향이 동전 던지기 — 캡션의 냄새 쪽 차이가 행동에서 가르는지 보는 자리.
09-29(train-v6 · runs/scorecard-20260928-124155): 진짜 0.558 · s0 0.508 · s1 0.580 · s2 0.556 · no_circuit 0.690 · text 0.097.

사용: uv run python scripts/50-analysis/decision_turns.py runs/scorecard-20260928-124155
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moai_echomind_fly import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument(
        "--after",
        type=int,
        default=None,
        help="이 결정 번호부터(기본 = 지금 task 의 sniff_decisions)",
    )
    a = ap.parse_args()
    after = (
        a.after
        if a.after is not None
        else int(config.load("task")["episode"].get("sniff_decisions", 0))
    )
    card = json.loads((Path(a.folder) / "scorecard.json").read_text(encoding="utf-8"))
    files = sorted(Path(a.folder).glob("*.decisions.jsonl"))
    if not files:
        print("결정 기록이 없다 — scorecard-v3(2026-09-26 e069579) 이전 성적표")
        return
    print(
        f"{a.folder} · 결정 {after} 부터 · 판정 {card.get('judgement', {}).get('rule')} · "
        f"vector_beats_rewired {card.get('vector_beats_rewired')}"
    )
    for f in files:
        name = f.name.removesuffix(".decisions.jsonl")
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        lr = [r for r in rows if r["d"] >= after and r["teacher"].get("turn") in ("left", "right")]
        same = sum(r["action"].get("turn") == r["teacher"]["turn"] for r in lr)
        opp = sum(
            r["action"].get("turn") in ("left", "right")
            and r["action"]["turn"] != r["teacher"]["turn"]
            for r in lr
        )
        reach = card["configs"].get(name, {}).get("reach_rate")
        print(
            f"  {name:11} 결정 {len(rows):4} · 교사가 돌라 한 결정 {len(lr):3} · 같은 쪽 {same / max(len(lr), 1):.3f}"
            f" · 반대쪽 {opp / max(len(lr), 1):.3f} · 먹이 도착 {reach}"
        )


if __name__ == "__main__":
    main()

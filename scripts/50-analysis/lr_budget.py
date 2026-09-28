"""냄새 좌우 차 예산 — 결정마다 두 센서의 상대 차가 얼마였나(가벼움 · 파일만).

① 자료에 기록된 두 센서 값의 상대 차 |L−R|/평균(자료가 모인 센서 기준 — fly-loop-v2 는 작은턱수염 · v4 는 더듬이)
② 더듬이 식 2·s·|sin 방위|/거리(s = 더듬이 간격 0.171 mm · 연구 노트 14 §3.1) — 과제를 바꿨을 때 어림.
M1b 문턱과 나란히: LN4 는 좌우 차 5~10 % 가 1 s 이상 일정해야 가른다(runs/odor-contrast-20260926-184048.json).
09-26: fly-loop-v2 를 더듬이 식으로 보면 중앙 0.58 % · 5 % 이상 11 % 뿐 → 과제를 바꿔야 한다는 근거.

사용: uv run python scripts/50-analysis/lr_budget.py fly-loop-v4 [train|eval] [--sep 0.171]
"""

from __future__ import annotations

import argparse

import numpy as np

from moai_echomind_fly import datasets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack")
    ap.add_argument("split", nargs="?", default="train")
    ap.add_argument("--sep", type=float, default=0.171, help="더듬이 간격 mm(FlyGym 실측)")
    a = ap.parse_args()
    _, rows = datasets.load(a.pack, a.split, "real")
    t = [r["truth"] for r in rows]
    rec = np.array(
        [
            abs(x["odor_left"] - x["odor_right"])
            / max((x["odor_left"] + x["odor_right"]) / 2, 1e-9)
            for x in t
        ]
    )
    d = np.array([x["food_distance_mm"] for x in t])
    az = np.radians([x["odor_azimuth_deg"] for x in t])
    est = 2 * a.sep * np.abs(np.sin(az)) / np.maximum(d, 0.5)
    side = np.abs(np.degrees(az)) > 10
    for name, v in (("기록된 센서", rec), (f"더듬이 식(s={a.sep})", est)):
        print(f"{a.pack} {a.split} · {name} · 좌우 차 중앙 {100 * np.median(v):.2f} %")
        for th in (0.02, 0.05, 0.10):
            m = v >= th
            print(
                f"  ≥ {th:.0%}: 결정 {m.mean():.1%} · 그중 왼/오른 칸(|방위| > 10°) {side[m].mean() if m.any() else 0:.0%}"
            )


if __name__ == "__main__":
    main()

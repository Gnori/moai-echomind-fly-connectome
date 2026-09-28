"""자료 묶음의 에피소드 모양 — 과제를 바꾸기 전에 본다(가벼움 · 파일만).

에피소드당 결정 수 · 결정당 먹이 거리 감소(걷는 속도) · 먹이 거리 구간별 결정 수와 방위 크기 · 물체 있는 결정.
09-26: fly-loop-v2 는 에피소드당 결정 중앙 10 · 결정당 3 mm · 교사가 금방 먹이 쪽으로 돌아 |방위| 중앙 8~18°
→ "먹이만 가깝게"는 에피소드가 2~3 결정으로 짧아진다는 판단(task-v4 에 냄새 맡기 구간을 넣은 까닭).

사용: uv run python scripts/50-analysis/episode_stats.py fly-loop-v4 [train|eval]
"""

from __future__ import annotations

import argparse

import numpy as np

from moai_echomind_fly import datasets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack")
    ap.add_argument("split", nargs="?", default="train")
    a = ap.parse_args()
    _, rows = datasets.load(a.pack, a.split, "real")
    eps: dict[int, list[dict]] = {}
    for r in rows:
        eps.setdefault(r["seed"], []).append(r)
    n = [len(v) for v in eps.values()]
    dd = []
    for v in eps.values():
        v = sorted(v, key=lambda x: x["decision"])
        dd += [
            p["truth"]["food_distance_mm"] - q["truth"]["food_distance_mm"]
            for p, q in zip(v, v[1:])
        ]
    d = np.array([r["truth"]["food_distance_mm"] for r in rows])
    az = np.abs([r["truth"]["odor_azimuth_deg"] for r in rows])
    print(
        f"{a.pack} {a.split} · 에피소드 {len(eps)} · 결정 {len(rows)} · 에피소드당 결정 중앙 {np.median(n):.0f}"
        f" · 결정당 거리 감소 중앙 {np.median(dd):.2f} mm"
    )
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        m = (d >= lo) & (d < hi)
        if m.any():
            print(
                f"  먹이 {lo}-{hi if hi < 1e9 else '∞'} mm · 결정 {m.sum()} · |방위| 중앙 {np.median(az[m]):.0f}°"
            )
    print(f"  물체 있는 결정 {sum(r['truth']['looming_present'] for r in rows)} / {len(rows)}")


if __name__ == "__main__":
    main()

"""좌우 차가 큰 결정만으로 냄새 쪽 선형 검사(사후 · 탐색 · 가벼움 · CPU 수 초).

좌우 차(기록 센서) ≥ 문턱인 결정만 학습 · 평가에 써서 왼쪽/오른쪽 두 갈래를 맞힌다(앞 칸은 빠진다).
전체로 배운 분류기를 부분에만 채점하면 "앞"이 많은 학습이 치우치므로 부분으로 다시 배운다.
09-26 M3(fly-loop-v2-ln): LN4 ≥ 5 % 0.594/찍기 0.575 · ≥ 10 % 0.672/0.627 — 흔들림 안 → "과제가 먼저" 판단.
⚠️ 결과를 본 뒤의 탐색이라 판정 근거로 쓰지 않는다(판정은 odor-windows · stats).

사용: uv run python scripts/50-analysis/subset_probe.py fly-loop-v4 --channel smell_ln --min-rel 0.05,0.10 --windows 1,3
"""

from __future__ import annotations

import argparse

import numpy as np

from moai_echomind_fly import datasets
from moai_echomind_fly.loop.linear_probe import ProbeSet
from moai_echomind_fly.loop.odor_windows import window_mean


def _rel(r: dict) -> float:
    t = r["truth"]
    return abs(t["odor_left"] - t["odor_right"]) / max((t["odor_left"] + t["odor_right"]) / 2, 1e-9)


def _side(r: dict) -> str:
    return "왼" if r["truth"]["odor_azimuth_deg"] > 0 else "오"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack")
    ap.add_argument("--channel", default="smell_ln")
    ap.add_argument("--min-rel", default="0.05,0.10")
    ap.add_argument("--windows", default="1,3")
    a = ap.parse_args()
    brains = ["real", *datasets.rewired_brains(a.pack)]
    for th in [float(x) for x in a.min_rel.split(",")]:
        for brain in brains:
            ftr, rtr = datasets.load(a.pack, "train", brain)
            fte, rte = datasets.load(a.pack, "eval", brain)
            mtr = np.array([_rel(r) >= th for r in rtr])
            mte = np.array([_rel(r) >= th for r in rte])
            ytr = [_side(r) for r, m in zip(rtr, mtr) if m]
            yte = [_side(r) for r, m in zip(rte, mte) if m]
            for k in [int(x) for x in a.windows.split(",")]:
                xtr = window_mean(ftr[a.channel].astype(float), rtr, k)[mtr]
                xte = window_mean(fte[a.channel].astype(float), rte, k)[mte]
                c = ProbeSet(xtr, xte).field(ytr, yte)
                print(
                    f"좌우 차 ≥ {th:.0%} · {brain:11} · {a.channel} 누적 {k}: 학습 {mtr.sum()} · 평가 {mte.sum()}"
                    f" · 정답 {c['accuracy']:.3f} · 찍기 {c['baseline']:.3f}"
                )


if __name__ == "__main__":
    main()

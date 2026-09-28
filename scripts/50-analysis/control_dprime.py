"""양성 대조(M1 · inspect odor-control) 결과의 d′ 와 한 개 빼고 문턱 정답률(가벼움 · 파일만).

M1 의 부호 규칙("자극한 쪽이 더 높나")은 바탕이 한쪽으로 치우친 뉴런(LN4)에서 0.5 가 된다(09-26 실측).
그래서 발화율 좌우 차(L−R)가 자극한 쪽에 따라 얼마나 갈리는지(d′)와, 두 평균의 가운데를 문턱으로 한 정답률을 본다.
09-26: PN d′ 7.7~15.5(차 0.35~1.05 Hz) · LN4 d′ 10.5~36.2(차 5~16 Hz 이동) · 둘 다 문턱 정답률 1.0.

사용: uv run python scripts/50-analysis/control_dprime.py runs/odor-control-20260926-175017.json [뇌]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from moai_echomind_fly.inspect import _loo_threshold_acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("brain", nargs="?", default="real")
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    trials = d["brains"][a.brain]["trials"]
    groups = [k for k in trials[0] if isinstance(trials[0][k], dict)]
    for g in groups:
        for rate in sorted({t["rate_hz"] for t in trials}):
            lf = np.array(
                [
                    t[g]["L_hz"] - t[g]["R_hz"]
                    for t in trials
                    if t["rate_hz"] == rate and t["side"] == "L"
                ]
            )
            rt = np.array(
                [
                    t[g]["L_hz"] - t[g]["R_hz"]
                    for t in trials
                    if t["rate_hz"] == rate and t["side"] == "R"
                ]
            )
            if len(lf) < 2 or len(rt) < 2:
                print(f"{g:6} {rate:>5.0f} Hz · 반복이 1 이라 d′ 를 못 잰다")
                continue
            sd = np.sqrt((lf.var(ddof=1) + rt.var(ddof=1)) / 2)
            dp = (lf.mean() - rt.mean()) / sd if sd else float("nan")
            print(
                f"{g:6} {rate:>5.0f} Hz · 왼쪽 자극 L−R {lf.mean():+.2f}±{lf.std(ddof=1):.2f} · 오른쪽 자극 "
                f"{rt.mean():+.2f}±{rt.std(ddof=1):.2f} · d′ {dp:.1f} · 문턱 정답률 {_loo_threshold_acc(lf, rt):.3f}"
            )


if __name__ == "__main__":
    main()

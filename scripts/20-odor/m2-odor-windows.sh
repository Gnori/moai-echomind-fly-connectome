#!/usr/bin/env bash
# 20 · M2 여러 창 선형 검사(가벼움 · CPU 수 분 · 새 시뮬레이션 없음) — 냄새 채널을 직전 결정 1·3·5·10 개 누적 + 시간 차.
# 규칙은 자료 판에 따라 자동(stats-v2 · task-v4 자료는 stats-v3). 냄새 맡기 구간만 보려면 --max-decision 4.
# 사용: scripts/20-odor/m2-odor-windows.sh <묶음> [채널…]
#   예: scripts/20-odor/m2-odor-windows.sh fly-loop-v4 smell_ln smell
# 09-26~27 결과: v2(옛 과제) 누적해도 찍기 수준 · v2-ln(LN4) 찍기 수준 · v4(task-v4) LN4 냄새 맡기 4 누적 0.725(재배선 0.525 · 찍기 0.490)
source "$(dirname "$0")/../common.sh"
DS=${1:?묶음}; shift; CHS=${*:-smell_ln smell}
for ch in $CHS; do
  step 0 odor-windows --dataset "$DS" --channel "$ch"
  step 0 odor-windows --dataset "$DS" --channel "$ch" --windows 1,2,4 --max-decision 4
done
step 0 linear-probe --dataset "$DS" --judge-only
say "결과 runs/odor-windows-*.json · runs/linear-probe-*.json(로그 $LOG)"

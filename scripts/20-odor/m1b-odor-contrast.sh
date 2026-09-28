#!/usr/bin/env bash
# 20 · M1b 약한 좌우 차(무거움 · 뇌만) — 공통 40 Hz 에서 좌우 차 c 를 주고 창 길이별로 ORN · PN · LN4 가 좌우를 가르나.
# 판정(결과 전에 정함): 한 개 빼고 문턱 정답률 ≥ 0.75. 09-26: 240 회 · 2,005 s · runs/odor-contrast-20260926-184048.json
#   → ORN 2 % 부터 · PN 은 20 % · 3 s 에서야 · LN4 는 5~10 % · 1 s 이상.
# 사용: scripts/20-odor/m1b-odor-contrast.sh [좌우차목록] [창목록ms] [반복]
source "$(dirname "$0")/../common.sh"
C=${1:-0.02,0.05,0.1,0.2}; W=${2:-300,1000,3000}; R=${3:-10}
no_lock
step 40 inspect odor-contrast --contrasts "$C" --windows-ms "$W" --reps "$R"
say "결과 $(ls -t runs/odor-contrast-*.json | head -1)"

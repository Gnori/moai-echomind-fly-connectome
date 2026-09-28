#!/usr/bin/env bash
# 20 · M1 양성 대조(무거움 · 뇌만 · 연구 노트 14 §7) — ORN_DM1 · VA2 를 한쪽만 구동해 PN · LN4 · DNa02 좌우 발화 차.
# 사용: scripts/20-odor/m1-odor-control.sh [--small]
#   --small : 진짜만 · 40 Hz · 반복 1(시행 2 · 약 10 s · 1.7 GB) — 새 종류는 작게 먼저
#   기본    : 16 · 40 · 150 Hz × 좌우 × 반복 10 × 1 s · 진짜 + 재배선 s0 = 120 회(09-26 실측 725 s)
# 결과: runs/odor-control-<시각>.json · d′ 와 문턱 정답률은 scripts/50-analysis/control_dprime.py 가 이어서 찍는다.
# 09-26 결과: PN 부호 1.000(차 0.35~1.05 Hz · 바탕 26 Hz) · LN4 부호 0.5(바탕 오른쪽 치우침 → 문턱으로는 d′ 10~36) · DNa02 0 Hz
source "$(dirname "$0")/../common.sh"
no_lock
if [ "${1:-}" = "--small" ]; then step 40 inspect odor-control --rates 40 --reps 1 --rewired-seeds ""
else step 40 inspect odor-control --rates 16,40,150 --reps 10 --rewired-seeds 0; fi
F=$(ls -t runs/odor-control-*.json | head -1); say "결과 $F"
run "$PY" scripts/50-analysis/control_dprime.py "$F"

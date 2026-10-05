#!/usr/bin/env bash
# 20 · 가소성 대조 다섯(F-007 · I v2 §8.3) — 짝짓기 · 짝 안 지은 냄새 · 어긋난 보상 · 처벌 DAN · 구획 표 섞기 · 가소성 끔.
#   무거움(짧음 · 뇌 전체 · 조건 × 씨앗). 기준은 결과 전 config/plasticity.yaml probe_controls.
# 사용: scripts/20-odor/plasticity-controls.sh [진짜 학습 폴더] [재배선 학습 폴더] [씨앗들]
#   → runs/plasticity-controls-<뇌>-<시각>.json
source "$(dirname "$0")/../common.sh"
REAL=${1:-runs/fit-kc-mbon-a-sim-v3-real-20261005-114119}
REW=${2:-runs/fit-kc-mbon-a-sim-v3-rewired-s0-20261005-170455}
SEEDS=${3:-0,1,2,3,4}
no_lock
step 40 plasticity-controls --cell-from "$REAL" --brain real --seeds "$SEEDS"
step 40 plasticity-controls --cell-from "$REW" --brain rewired-s0 --seeds "$SEEDS"

#!/usr/bin/env bash
# 20 · η 다시 보정 → 대조 다섯 다시(10-13 · 결과 전 잠금 config/plasticity.yaml eta_calibration calib-a) · 가벼움(약 25 분)
#   ① 짝짓기 · CS+ 만 · 씨앗 10~14 · η 격자 → 고름 ② 고른 η 로 대조 다섯 · 씨앗 0~4 · 진짜 · 재배선 · CS− 는 10-13 규칙으로 고른 것 그대로
source "$(dirname "$0")/../common.sh"
REAL=${1:-runs/fit-kc-mbon-al-sim-v3-real-20261013-071105}
REW=${2:-runs/fit-kc-mbon-al-sim-v3-rewired-s0-20261013-123831}
CSM=ORN_DA1,ORN_DC1,ORN_DP1l,ORN_VA1v,ORN_VC5,ORN_VM6l
no_lock
step 40 plasticity-eta-calibrate --cell-from "$REAL"
ETA=$("$PY" -c "import json,glob;print(json.load(open(sorted(glob.glob('runs/plasticity-eta-calibrate-*.json'))[-1]))['pick'])")
say "고른 η $ETA"
step 40 plasticity-controls --cell-from "$REAL" --brain real --seeds 0,1,2,3,4 --cs-minus "$CSM" --eta "$ETA"
step 40 plasticity-controls --cell-from "$REW" --brain rewired-s0 --seeds 0,1,2,3,4 --cs-minus "$CSM" --eta "$ETA"

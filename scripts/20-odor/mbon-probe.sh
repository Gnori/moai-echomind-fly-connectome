#!/usr/bin/env bash
# 20 · MBON 침묵 진단(F-007 · M3 뒤) — MBON 마다 입력이 문턱에 얼마나 못 미치나. 무거움(짧음 · 뇌 전체 1 s × 경우 수)
# 사용: scripts/20-odor/mbon-probe.sh [진짜 회로학습 폴더] [재배선 회로학습 폴더]   → runs/mbon-probe-<시각>.json
#   경우 셋: 진짜 학습 전 · 진짜 학습 뒤 · 재배선 s0 학습 뒤(섞은 배선과 나란히)
#   판정 기준(결과 전 · config/plasticity.yaml mbon_probe): fires · inhibition · no_kc_input · weak_kc(+ KC 필요 배수)
source "$(dirname "$0")/../common.sh"
REAL=${1:-runs/fit-kc-sparse-b-sim-v3-real-20261003-141045}
REW=${2:-runs/fit-kc-sparse-b-sim-v3-rewired-s0-20261003-173311}
no_lock
step 40 mbon-probe --case real-before=real --case "real-fit=real:$REAL" \
  --case "rewired-fit=rewired-s0:$REW"

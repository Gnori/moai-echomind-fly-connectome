#!/usr/bin/env bash
# 20 · 짝짓기에서 CS− 가 왜 같이 약해지나 — 보상 구획 KC 입력의 CS+ · CS− 겹침 분해(F-007 · 10-06 · 탐색 · 판정 아님).
#   짝짓기 조건만 · 같은 CS− · 씨앗 0~4 · --detail. 무거움(짧음 · 뇌 전체).
# 사용: scripts/20-odor/plasticity-overlap.sh [진짜 학습 폴더] [재배선 학습 폴더] [CS−]
#   → runs/plasticity-controls-<뇌>-<시각>.json 의 rows[].split
source "$(dirname "$0")/../common.sh"
REAL=${1:-runs/fit-kc-mbon-b-sim-v3-real-20261006-071233}
REW=${2:-runs/fit-kc-mbon-b-sim-v3-rewired-s0-20261006-105313}
CSM=${3:-ORN_DL5,ORN_DM5,ORN_DM6,ORN_VA1v,ORN_VA3,ORN_VM6m}
no_lock
step 40 plasticity-controls --cell-from "$REAL" --brain real --seeds 0,1,2,3,4 --conditions paired --cs-minus "$CSM" --detail
step 40 plasticity-controls --cell-from "$REW" --brain rewired-s0 --seeds 0,1,2,3,4 --conditions paired --cs-minus "$CSM" --detail

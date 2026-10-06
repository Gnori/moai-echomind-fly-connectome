#!/usr/bin/env bash
# 20 · KC 부호 겹침이 회로 학습 탓인가 — 학습 전 · kc-sparse-c · kc-mbon-a 에서 같은 겹침 분해(F-007 · 10-06 · 탐색).
#   plasticity-overlap.sh 와 같은 CS− · 짝짓기 · 씨앗 0~4 · --detail. 무거움(짧음 · 뇌 전체 · 한 번에 1~2 분).
# 사용: scripts/20-odor/plasticity-overlap-history.sh   → runs/plasticity-controls-<뇌>-*.json rows[].split
source "$(dirname "$0")/../common.sh"
CSM=${1:-ORN_DL5,ORN_DM5,ORN_DM6,ORN_VA1v,ORN_VA3,ORN_VM6m}
no_lock
for pair in "none real" "none rewired-s0" \
            "runs/fit-kc-sparse-c-sim-v3-real-20261005-011859 real" \
            "runs/fit-kc-sparse-c-sim-v3-rewired-s0-20261005-054938 rewired-s0" \
            "runs/fit-kc-mbon-a-sim-v3-real-20261005-114119 real" \
            "runs/fit-kc-mbon-a-sim-v3-rewired-s0-20261005-170455 rewired-s0"; do
  set -- $pair
  step 40 plasticity-controls --cell-from "$1" --brain "$2" --seeds 0,1,2,3,4 --conditions paired --cs-minus "$CSM" --detail
done

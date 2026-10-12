#!/usr/bin/env bash
# 20 · 촉각엽 고침 + PN→KC 배율 뒤 짝짓기 특이성 작은 시험(10-12 · 가능성 먼저 · 판정 아님 · 씨앗 0~2 · 약 2 분씩)
source "$(dirname "$0")/../common.sh"
CSM=ORN_DL5,ORN_DM5,ORN_DM6,ORN_VA1v,ORN_VA3,ORN_VM6m
no_lock
for d in "${@:-runs/al-kc-x0.5-g1.5 runs/al-kc-x0.25-g2}"; do
  for x in $d; do
    step 40 plasticity-controls --cell-from "$x" --brain real --seeds 0,1,2 --conditions paired --cs-minus "$CSM" --detail
  done
done

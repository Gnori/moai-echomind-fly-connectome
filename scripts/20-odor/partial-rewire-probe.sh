#!/usr/bin/env bash
# 20 · 부분 섞기 대조 가능성(10-13 · 탐색 · 판정 아님) — PN→KC · KC→MBON 만 섞은 뇌에 진짜 kc-mbon-al 매개변수를 그대로 얹어
#   짝짓기 + 분해(η 0.002 · 씨앗 0~2 · CS− 는 10-13 규칙으로 고른 것). MBON 이 켜져 비교가 되나 · 특이성이 어떻게 되나.
source "$(dirname "$0")/../common.sh"
CELL=${1:-runs/fit-kc-mbon-al-sim-v3-real-20261013-071105}
CSM=ORN_DA1,ORN_DC1,ORN_DP1l,ORN_VA1v,ORN_VC5,ORN_VM6l
no_lock
for b in pnkc-s0 kcmbon-s0; do
  step 40 plasticity-controls --cell-from "$CELL" --brain "$b" --seeds 0,1,2 --conditions paired --cs-minus "$CSM" --eta 0.002 --detail
done

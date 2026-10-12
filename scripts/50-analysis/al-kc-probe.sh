#!/usr/bin/env bash
# 50 · 촉각엽 고침 → KC 겹침 · 짝짓기 대조 작은 시험(10-12 · 가능성 먼저)
#   ① 배율마다 KC 켜짐 · 냄새끼리 KC 겹침(약 1 분씩) ② 배율마다 짝짓기 + CS− 분해(plasticity-controls · 씨앗 0~2 · 약 1 분씩)
source "$(dirname "$0")/../common.sh"
F=${1:-1.0,0.5,0.25}
CSM=ORN_DL5,ORN_DM5,ORN_DM6,ORN_VA1v,ORN_VA3,ORN_VM6m
no_lock
run "$PY" scripts/50-analysis/al_kc_probe.py "$F"
for f in ${F//,/ }; do
  step 40 plasticity-controls --cell-from "runs/al-kc-x$(python3 -c "print(f'{float(\"$f\"):g}')")" --brain real --seeds 0,1,2 --conditions paired --cs-minus "$CSM" --detail
done

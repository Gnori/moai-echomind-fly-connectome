#!/usr/bin/env bash
# 20 · 가치 읽기 A1(10-14 · 결과 전 잠금 valence_readout vr-a) — 몸 없이. 진짜 대조 다섯 · 섞은 갈래 여섯(짝짓기) · 씨앗 10 · --detail → 판정.
#   가벼움(약 30 분).
source "$(dirname "$0")/../common.sh"
G() { "$PY" -c "from moai_echomind_fly import config; c=config.load('plasticity')['valence_readout']; print($1)"; }
CELL=$(G "c['cell_from']"); ETA=$(G "c['eta']"); CSM=$(G "','.join(c['cs_minus'])"); SEEDS=$(G "','.join(map(str,c['seeds']))")
ARMS=$(G "' '.join(b for v in c['arms'].values() for b in v)")
no_lock
step 40 plasticity-controls --cell-from "$CELL" --brain real --seeds "$SEEDS" --cs-minus "$CSM" --eta "$ETA" --detail
for b in $ARMS; do
  step 40 plasticity-controls --cell-from "$CELL" --brain "$b" --seeds "$SEEDS" --conditions paired --cs-minus "$CSM" --eta "$ETA" --detail
done
run "$PY" scripts/50-analysis/valence_judge.py

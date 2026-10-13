#!/usr/bin/env bash
# 20 · 부분 섞기 정식 판정(10-13 · 결과 전 잠금 partial_rewire_test pr-a) — 뇌 7 × 짝짓기 × 씨앗 10 → 판정. 가벼움(약 20~30 분).
source "$(dirname "$0")/../common.sh"
G() { "$PY" -c "from moai_echomind_fly import config; c=config.load('plasticity')['partial_rewire_test']; print($1)"; }
CELL=$(G "c['cell_from']"); ETA=$(G "c['eta']"); CSM=$(G "','.join(c['cs_minus'])"); SEEDS=$(G "','.join(map(str,c['seeds']))")
BRAINS=$(G "' '.join(['real']+[b for v in c['arms'].values() for b in v])")
no_lock
for b in $BRAINS; do
  step 40 plasticity-controls --cell-from "$CELL" --brain "$b" --seeds "$SEEDS" --conditions paired --cs-minus "$CSM" --eta "$ETA"
done
run "$PY" scripts/50-analysis/partial_rewire_judge.py

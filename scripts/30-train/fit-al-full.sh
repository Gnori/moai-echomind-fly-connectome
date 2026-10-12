#!/usr/bin/env bash
# 30 · kc-mbon-al 본 학습 → 대조(10-13 · 요구자 "응 진행해") — 무거움(약 9 h).
#   ① 회로 학습 50 세대 진짜 → 재배선 s0(config/fit.yaml kc-mbon-al · 결과 전 a7302b6)
#   ② CS− 고르기(진짜 · 규칙 cs_minus_screen 그대로) ③ 대조 다섯 · 씨앗 0~4 · 진짜 · 재배선(probe_controls 그대로 · η 0.001)
source "$(dirname "$0")/../common.sh"
no_lock
step 40 fit --scope kc-mbon-al --method cmaes --brain real
REAL=$(ls -td runs/fit-kc-mbon-al-sim-v3-real-* | head -1)
step 40 fit --scope kc-mbon-al --method cmaes --brain rewired-s0
REW=$(ls -td runs/fit-kc-mbon-al-sim-v3-rewired-s0-* | head -1)
say "학습 폴더 진짜 $REAL · 재배선 $REW"
step 40 fit-eval --run "$REAL"
step 40 fit-eval --run "$REW"
step 40 plasticity-cs-screen --cell-from "$REAL"
PICK=$("$PY" -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/plasticity-cs-screen-*.json'))[-1]));p=d['pick'];print(','.join(p['types']) if p else '')")
if [ -z "$PICK" ]; then say "고를 짝 안 지은 냄새 없음 — 설정 기본 CS−"; CS="--cs-minus="; else say "고른 CS− $PICK"; CS="--cs-minus=$PICK"; fi
step 40 plasticity-controls --cell-from "$REAL" --brain real --seeds 0,1,2,3,4 "$CS"
step 40 plasticity-controls --cell-from "$REW" --brain rewired-s0 --seeds 0,1,2,3,4 "$CS"

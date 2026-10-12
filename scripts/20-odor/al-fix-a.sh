#!/usr/bin/env bash
# 20 · al-fix-a 정식 시험(10-12 · 결과 전 잠금 7bf33f2) — ① PN→KC 고르기(진짜 · 재배선) ② 짝 안 지은 냄새 고르기(규칙 그대로)
#   ③ 대조 다섯 · 씨앗 0~4 · 진짜 · 재배선. 가벼움(약 20~30 분).
source "$(dirname "$0")/../common.sh"
no_lock
wait_free 40
run "$PY" scripts/50-analysis/al_fix_select.py real,rewired-s0
step 40 plasticity-cs-screen --cell-from runs/al-fix-a-real
PICK=$("$PY" -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/plasticity-cs-screen-*.json'))[-1]));p=d['pick'];print(','.join(p['types']) if p else '')")
if [ -z "$PICK" ]; then say "고를 짝 안 지은 냄새 없음 — 설정 기본 CS− 로 대조(CS− 는 잴 수 없을 수 있음)"; CS=(); else say "고른 CS− $PICK"; CS=(--cs-minus "$PICK"); fi
step 40 plasticity-controls --cell-from runs/al-fix-a-real --brain real --seeds 0,1,2,3,4 "${CS[@]}"
step 40 plasticity-controls --cell-from runs/al-fix-a-rewired-s0 --brain rewired-s0 --seeds 0,1,2,3,4 "${CS[@]}"

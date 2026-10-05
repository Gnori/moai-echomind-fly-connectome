#!/usr/bin/env bash
# 20 · 짝 안 지은 냄새(CS−) 고르기 → 고른 냄새로 짝짓기 조건만 다시(F-007 · 10-06). 무거움(짧음).
#   고르는 규칙은 결과 전 config/plasticity.yaml probe_controls.cs_minus_screen(짝짓기 전 반응만 본다).
# 사용: scripts/20-odor/plasticity-cs-screen.sh [진짜 학습 폴더]   → runs/plasticity-cs-screen-*.json
#   고른 냄새가 있으면 이어서 plasticity-controls --conditions paired --cs-minus <고른 것>(씨앗 0~4)
source "$(dirname "$0")/../common.sh"
REAL=${1:-runs/fit-kc-mbon-a-sim-v3-real-20261005-114119}
no_lock
step 40 plasticity-cs-screen --cell-from "$REAL"
PICK=$("$PY" -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/plasticity-cs-screen-*.json'))[-1]));p=d['pick'];print(','.join(p['types']) if p else '')")
if [ -z "$PICK" ]; then say "고를 냄새 없음 — 기록만"; exit 0; fi
say "고른 CS− $PICK"
step 40 plasticity-controls --cell-from "$REAL" --brain real --seeds 0,1,2,3,4 --conditions paired --cs-minus "$PICK"

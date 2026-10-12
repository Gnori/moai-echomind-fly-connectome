#!/usr/bin/env bash
# 20 · 촉각엽 고침 + KC 다시 맞춤(작은 학습) 뒤 짝짓기 특이성 가능성(10-13 · 판정 아님) — CS− 고르기(규칙 그대로) → 짝짓기 + 분해(씨앗 0~2)
source "$(dirname "$0")/../common.sh"
CELL=${1:?회로 학습 폴더}
no_lock
step 40 plasticity-cs-screen --cell-from "$CELL"
PICK=$("$PY" -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/plasticity-cs-screen-*.json'))[-1]));p=d['pick'];print(','.join(p['types']) if p else '')")
if [ -z "$PICK" ]; then say "고를 짝 안 지은 냄새 없음"; exit 0; fi
say "고른 CS− $PICK"
step 40 plasticity-controls --cell-from "$CELL" --brain real --seeds 0,1,2 --conditions paired --cs-minus "$PICK" --detail

#!/usr/bin/env bash
# 60 · V0b(R-1177) — 어두운 원판(다가옴 · 물러남 · 어두워지기만) → T4/T5 + 소엽 기둥 → LC4 · LPLC2 · GF.
#   설정 v0b 에 '미정'이 있으면 거부(끝 코드 2). 인자는 그대로 넘김. CPU 약 30 분(540 번).
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v0b_disk.py "$@"

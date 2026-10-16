#!/usr/bin/env bash
# 60 · V1(R-1177) — 렌더러 그림(arena-a · 몸 돎 · 사람 다가옴) → 앞단 v2 → 뇌 · 진짜 · 자리 섞기 5 · 재배선 3 · 씨앗 10.
#   결과 전 잠금 eyeworld.yaml v1-a. 인자 그대로(--frames <폴더> 로 그림 다시 쓰기). 헤드리스 크롬 하나 · CPU 약 30~40 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v1_check.py "$@"

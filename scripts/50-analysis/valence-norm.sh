#!/usr/bin/env bash
# 50 · 가치 읽기 A1 정규화 탐색(10-14 · 결과 뒤라 판정 아님 · 저장된 MBON 발화로 계산만 · 몇 초) → runs/valence-norm-*.json
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/50-analysis/valence_norm.py "$@"

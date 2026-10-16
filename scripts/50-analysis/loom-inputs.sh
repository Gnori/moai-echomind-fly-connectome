#!/usr/bin/env bash
# 50 · 다가옴 길 입력 구성(10-16 · 조회만) — LPLC2 · LC4 · GF 입력 종류 · 좌표 비율. V0b 설계 재료. 수십 초.
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/50-analysis/loom_inputs.py

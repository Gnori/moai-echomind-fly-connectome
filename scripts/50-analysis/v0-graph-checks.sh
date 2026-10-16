#!/usr/bin/env bash
# 50 · V0 전 연결 지도 확인(10-16 · 조회만 · 연구 노트 17 §8.4 ①~④) — T4/T5 좌표 · VS/HS 입력 · HS/VS → DNa02 · LPLC2/LC4 의 T4/T5 몫. 수십 초.
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/50-analysis/v0_graph_checks.py

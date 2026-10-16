#!/usr/bin/env bash
# 50 · 소엽 기둥 뉴런 입구 재기(10-16 · 가벼움 · 판정 아님) — Tm5Y · Tm20 · Y3 · TmY5a · Tm37 · Tm4 · T2 · TmY3 · Tm2 · Tm3
#   두 눈 · 국소(앞-왼 30° · 20°) · 섞음 둘 → LC4 · LPLC2 · GF · DNp02 · 진짜 대 섞은 배선 s0~s2. CPU 약 5 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/lobula_entry.py

#!/usr/bin/env bash
# 50 · 눈 → 뇌 길 재기(10-16 · 가벼움 · 판정 아님) — 시각엽 층마다 켜서 LC/LPLC · HS/VS · DNa02 · GF 까지 가나 · 진짜 대 섞은 배선 s0~s2 · 왼눈만. CPU 몇 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/vision_path.py

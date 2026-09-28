#!/usr/bin/env bash
# 90 · 마무리(가벼움) — 학습 · 평가를 새로 돌린 뒤 한 번. 대장용 sha256 표 · 검사 · 백업.
# 사용: scripts/90-wrap/finalize.sh <판>
#   예: scripts/90-wrap/finalize.sh train-v6
# 남는 일(손으로): 학습 안내서 10장 대장 한 줄 · R(인수 기준) · 인수인계 · doc 세션에 문서 반영 일감(CLAUDE.md 코드 변경 시 문서 갱신 표).
source "$(dirname "$0")/../common.sh"
V=${1:?판}
run "$PY" scripts/50-analysis/ledger_hashes.py "$V"
say "위 표를 10장 1.3 에 옮긴다 · 전체 표 $(ls -t runs/scripts-ledger-*.md 2>/dev/null | head -1)"
run "$ROOT/.venv/bin/ruff" check src tests scripts
run "$ROOT/.venv/bin/pytest" -q -m "not heavy"
step 0 backup
say "USB(JKL-USB)가 없으면 건너뛴다 — 꽂은 뒤 다시: $FC backup"

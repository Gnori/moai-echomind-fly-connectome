#!/usr/bin/env bash
# 00 · 준비 점검(가벼움) — 무거운 실행 전에 한 번. 도는 것 · 잠금 · 메모리 · Ollama · 대시보드 · 코드 검사.
# 사용: scripts/00-check/check.sh [--no-tests]
source "$(dirname "$0")/../common.sh"
say "메모리 여유 $(free_pct)% (평소 40 % · Qwen 단계 63 % · 투영기 학습 40 % — config/guard.yaml)"
if [ -f runs/.heavy.lock ]; then say "⚠️ 잠금 있음: $(cat runs/.heavy.lock)"; else say "잠금 없음"; fi
say "도는 무거운 실행:"; pgrep -fl "fly-connectome (train|collect|replay|scorecard|eval|inspect|odor)" | cut -c1-160 | tee -a "$LOG" || true
command -v ollama >/dev/null && { say "Ollama 에 올라 있는 모델:"; ollama ps | tee -a "$LOG"; }
if pgrep -f "fly-connectome dashboard --serve" >/dev/null; then say "대시보드 켜짐 http://127.0.0.1:8770/ (설정을 바꿨으면 재시작)"
else say "대시보드 꺼짐 — 켜기: nohup $FC dashboard --serve 8770 --every 30 > runs/dashboard/serve.log 2>&1 &"; fi
say "GPU 를 쓰는 다른 무거운 작업이 있으면(크롬 3D · 다른 세션 시험) 학습이 2~3 배 느려지고 GPU 오류(InnocentVictim)가 날 수 있다(09-27 실측)"
if [ "${1:-}" != "--no-tests" ]; then
  run "$ROOT/.venv/bin/ruff" check src tests scripts
  run "$ROOT/.venv/bin/pytest" -q -m "not heavy"
fi
say "점검 끝 · 로그 $LOG"

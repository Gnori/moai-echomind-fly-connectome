#!/usr/bin/env bash
# 30 · 압축 · 관문 · 학습 메모리 측정(관문은 가벼움 · train-probe 는 Qwen 을 올려 무거움 · 약 3 분)
# 관문을 통과한 채널만 투영기 입력이 된다(F-004 규칙 · 통과가 없으면 학습하지 않는다).
# 사용: scripts/30-train/gate-and-probe.sh <묶음>
#   예: scripts/30-train/gate-and-probe.sh fly-loop-v4   → 관문: 시각 · smell_ln · 공간 · 최대 31.3 GB · 3.4~6.4 s/걸음(09-27)
# 새 종류의 학습(입력 채널 · 토큰 수가 바뀜)은 train-probe 로 최대 메모리 · 걸음 시간을 먼저 잰다(CLAUDE.md).
source "$(dirname "$0")/../common.sh"
DS=${1:?묶음}
no_lock
step 0 compress fit --dataset "$DS"
step 0 compress gate --dataset "$DS"
"$PY" -c "import json;from moai_echomind_fly.loop import compress;g=json.load(open(compress.out_dir('$DS')/'gate.json'));print('관문 통과 채널',g['passed_channels'],'· 칸',g['passed_cells'])" | tee -a "$LOG"
step 63 train-probe --dataset "$DS" --steps 12
say "train-probe 기록 $(ls -t runs/train-probe-*.jsonl | head -1) — peak_gb 가 평탄한지 · 걸음당 초로 뇌당 시간을 잡는다"

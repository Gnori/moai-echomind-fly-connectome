#!/usr/bin/env bash
# 40 · MBON 회로 학습 뒤 확인 한 번에(F-006 · F-007 · 10-06) — 무거움(짧음 · 차례로)
#   ① fit-eval 진짜 · 재배선(일반화 · 도망 · 발톱 · 겹침) ② mbon-probe(앞 판 대 새 판 · 진짜 · 재배선)
#   ③ plasticity-cs-screen(짝 안 지은 냄새 고르기 · 규칙은 결과 전 config) ④ 고른 냄새로 대조 다섯(진짜 · 재배선)
# 사용: scripts/40-eval/after-fit-mbon.sh <진짜 새 폴더> <재배선 새 폴더> <진짜 앞 폴더> <재배선 앞 폴더>
source "$(dirname "$0")/../common.sh"
R=${1:?진짜 새}; N=${2:?재배선 새}; R0=${3:?진짜 앞}; N0=${4:?재배선 앞}
no_lock
step 40 fit-eval --run "$R"
step 40 fit-eval --run "$N"
run "$PY" -c "from moai_echomind_fly.mb_probe import overlap; import sys; print('겹침(Jaccard)', overlap(sys.argv[1], sys.argv[2]))" "runs/fit-eval-$(basename "$R").json" "runs/fit-eval-$(basename "$N").json"
step 40 mbon-probe --case "real-old=real:$R0" --case "real-new=real:$R" --case "rewired-old=rewired-s0:$N0" --case "rewired-new=rewired-s0:$N"
step 40 plasticity-cs-screen --cell-from "$R"
PICK=$("$PY" -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/plasticity-cs-screen-*.json'))[-1]));p=d['pick'];print(','.join(p['types']) if p else '')")
if [ -z "$PICK" ]; then say "고를 냄새 없음 — 대조는 설정의 CS− 로"; CSM=(); else say "고른 CS− $PICK"; CSM=(--cs-minus "$PICK"); fi
step 40 plasticity-controls --cell-from "$R" --brain real --seeds 0,1,2,3,4 "${CSM[@]}"
step 40 plasticity-controls --cell-from "$N" --brain rewired-s0 --seeds 0,1,2,3,4 "${CSM[@]}"

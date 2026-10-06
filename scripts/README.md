# scripts — 재현 · 개선용 스크립트 (단계별 · 종류별)

> 2026-09-29 요구자 *"필요하면 스크립트도 남겨서 · 단계별로 · 종류별로 · 우리의 자산을 남겨야 나중에 더 발전할 때 빠르게 하지"*.
> 이 폴더는 **그대로 다시 하거나 판을 바꿔 다시 하는** 입구다. 절차 설명 · 까닭 · 결과 해석은 학습 안내서 12장(재현 절차와 다음 작업) ·
> 명령 하나하나는 11장 · 결과 대장은 10장 · 판정 근거는 R-7469(`docs/requirements/`).
>
> 공통 규칙(`common.sh`): 한 번에 하나(`runs/.heavy.lock`) · 시작 전 메모리 여유 문턱 · **끝 코드를 바로 받아 실패면 멈춤**(파이프로 가리지 않음) ·
> 로그는 `runs/scripts-<이름>-<시각>.log`. 무거운 실행은 **요구자 확인 뒤**(CLAUDE.md).

## 1. 단계별 — 처음부터 끝까지 한 판을 돌리는 순서

| 단계 | 스크립트 | 무게 | 하는 일 | 실측(2026-09-26~29) |
|---|---|---|---|---|
| 00 | `00-check/check.sh [--no-tests]` | 가벼움 | 메모리 · 잠금 · 도는 것 · Ollama · 대시보드 · ruff · pytest | 수 초 |
| 10 | `10-data/collect.sh <묶음> <학습씨앗> <평가씨앗> <재배선씨앗> [일꾼]` | **무거움** | collect-probe → 학습 · 평가 수집 → 묶기 → 체크섬 | fly-loop-v4(200 + 200 · 뇌 4 · 일꾼 3) **3 h 36 m** |
| 10 | `10-data/replay.sh <원본> <새묶음> <재배선씨앗> [학습씨앗] [--retune]` | **무거움** | 원본과 같은지 검증(최대 차 0) → 몸 없이 뇌만 재생 → 묶기 | fly-loop-v2-ln(평가 40 + 학습 120 · 뇌 2) 34 m + 71 m |
| 20 | `20-odor/m4-body-check.sh [재배선]` | 짧음 | 과제를 바꾼 뒤 몸 확인(냄새 맡기 동안 서 있나 · 좌우 차 %) | 에피소드 2 · 약 2 분 |
| 30 | `30-train/gate-and-probe.sh <묶음>` | 관문 가벼움 · probe 무거움 | 압축 · 관문(통과 채널 = 투영기 입력) → train-probe(최대 메모리 · 걸음 시간) | 관문 수 분 · probe 약 3 분 · 31.3 GB |
| 30 | `30-train/train-all.sh [--train-only=…] [--skip-train]` | **무거움** | `bin/run_full_v6.sh` — 학습 4 → 채점 평가 8 → 판정 2 → 성적표 | 약 2 일(뇌당 5~10 h · 성적표 약 12 h) |
| 40 | `40-eval/eval-judge.sh <판> <묶음> [뇌…]` | **무거움(Qwen)** | 채점 평가 + 판정만 다시 | 약 2 h(1,708 표본) |
| 40 | `40-eval/scorecard.sh <판> [씨앗]` | **무거움(Qwen · 몸)** | 성적표만 다시 + 결정 기록 분석 | 씨앗 40 × 7 약 12 h |
| 40 | `40-eval/scorecard-tiered.sh <판> [씨앗] [--measure]` | **무거움(Qwen · 몸)** | 층 구조 성적표(R-1027) — 늘 부름 · fast-only · gated · 같은 호출률 무작위 · 주기 · 재배선 gated · fast-only · 끝나면 `useful_calls.py` | 아직 안 돌림 · `--measure` 로 씨앗 2 개 먼저 |
| 90 | `90-wrap/finalize.sh <판>` | 가벼움 | 대장용 sha256 표 · ruff · pytest · 백업 | 수 분 |

## 2. 종류별

### 2.1 측정 · 실험(냄새 좌우 — 연구 노트 14 M0~M4)

| 스크립트 | 무엇 | 결과 파일 | 09-26 결과(요지) |
|---|---|---|---|
| `20-odor/m1-odor-control.sh [--small]` | 양성 대조 — ORN 한쪽만 구동 | `runs/odor-control-*.json` | PN 부호 1.000(차 2~4 %) · LN4 d′ 10~36 · DNa02 0 Hz |
| `20-odor/m1b-odor-contrast.sh [c] [창] [반복]` | 약한 좌우 차 × 창 길이 | `runs/odor-contrast-*.json` | ORN 2 %~ · PN 20 %·3 s 에서야 · LN4 5~10 %·1 s 이상 |
| `20-odor/m2-odor-windows.sh <묶음> [채널…]` | 여러 창 선형 검사 + 전 채널 선형 검사 | `runs/odor-windows-*.json` · `linear-probe-*.json` | v2 · v2-ln 찍기 수준 · **v4 LN4 냄새 맡기 4 누적 0.725** |
| `20-odor/mbon-probe.sh [진짜 학습 폴더] [재배선 학습 폴더]` | MBON 침묵 진단 — 출처별 평균 탈분극 대 문턱(F-007) | `runs/mbon-probe-*.json` | 10-05: 학습 뒤 KC 가 αβ-m 하나에만(37 %) → MBON 67/97 KC 입력 0 · αβ MBON 은 APL 억제 · 재배선 ×250 |
| `20-odor/plasticity-probe.sh [냄새] [η] [학습 폴더] [씨앗들] [뇌]` | 가소성 먼저 재기 — 짝짓기 1 s 뒤 보상 구획 MBON CS+ 변화(F-007) | `runs/plasticity-probe-*.json` | 10-06 진짜 kc-mbon-a: η 0.001 에서 76 → 28(−63 % · 5/5) · η 0 76 → 84 · 재배선 MBON 침묵 |
| `20-odor/plasticity-controls.sh [진짜 폴더] [재배선 폴더] [씨앗들]` | 가소성 대조 다섯(짝짓기 · 어긋난 보상 · 처벌 DAN · 구획 표 섞기 · 끔 + CS−) | `runs/plasticity-controls-<뇌>-*.json` | 10-06 진짜: −64 % · 처벌 DAN 이면 처벌 구획 −65 % · 대조 +4 / −12 / +4 % · 재배선 MBON 침묵 |
| `40-eval/after-fit-mbon.sh <진짜 새> <재배선 새> <진짜 앞> <재배선 앞>` | MBON 회로 학습 뒤 확인 한 번에 — fit-eval 둘 · mbon-probe 넷 · CS− 고르기 · 대조 다섯(진짜 · 재배선) | 위 각 결과 파일 | 10-06 kc-mbon-b: MBON 30/97 대 0/97 · 대조 CS+ −37 % · CS− −40 %(냄새 특이성 없음) |
| `bin/run_m3.sh` · `bin/run_m4_pilot.sh` · `bin/run_m4_full.sh` | 그때 돈 순서 그대로(기록용) | — | 위 10 · 20 단계 스크립트로 같은 일을 한다 |

### 2.2 자료

| 스크립트 | 무엇 |
|---|---|
| `10-data/collect.sh` | 새 과제 · 채널 판으로 수집(몸 + 뇌) |
| `10-data/replay.sh` | 채널 · 시야 조율만 바꿀 때(뇌만 재생) |

### 2.3 학습 · 평가

| 스크립트 | 무엇 |
|---|---|
| `30-train/gate-and-probe.sh` | 학습 전 관문 · 메모리 |
| `30-train/train-all.sh` | 전체 순서(멈추면 `--train-only` 로 그 뇌부터 처음부터) |
| `40-eval/eval-judge.sh` · `40-eval/scorecard.sh` | 평가 · 성적표만 |

### 2.4 분석(가벼움 · 파일만 · 그때 명령줄에서 즉석으로 돌렸던 것을 남김)

| 스크립트 | 무엇 | 그때 쓴 결정 |
|---|---|---|
| `50-analysis/episode_stats.py <묶음> [나눔]` | 에피소드당 결정 · 걷는 속도 · 거리 구간별 방위 | "먹이만 가깝게는 에피소드가 너무 짧다" → 냄새 맡기 구간 |
| `50-analysis/lr_budget.py <묶음> [나눔] [--sep]` | 좌우 차 분포(기록 센서 · 더듬이 식) | v2 더듬이 기준 중앙 0.58 % · 5 % 이상 11 % → 과제 바꿈 |
| `50-analysis/subset_probe.py <묶음> …` | 좌우 차 큰 결정만 두 갈래 검사(사후 · 탐색) | M3 에서 흔들림 안 → "과제가 먼저" |
| `50-analysis/control_dprime.py <odor-control.json>` | M1 d′ · 문턱 정답률 | LN4 는 부호가 아니라 문턱으로 봐야 |
| `50-analysis/decision_turns.py <성적표 폴더>` | 교사와 같은 쪽으로 돈 비율(scorecard-v3 결정 기록) | v6: 진짜 0.558 · s0 0.508 · s1 0.580 · s2 0.556 |
| `50-analysis/ledger_hashes.py <판>` | 10장 대장 표(손실 · sha256 · 지문) | v6 대장 1.3 |
| `50-analysis/useful_calls.py <성적표 폴더>` | 층 구성에서 부른 결정이 쓸모 있었나(계기별 · 빠른 정책과 다름 · 교사 일치 LLM 대 빠른 정책) | R-1027 · 연구 노트 15 T3b |

모두 `uv run python scripts/50-analysis/<이름>.py --help` 로 옵션을 본다. 09-29 에 실제 자료로 돌려 그때 값이 그대로 나오는 것을 확인했다.

## 3. 자산 — 다시 쓰는 것들(git 밖은 백업 `~/Backups` + USB)

| 종류 | 자리 | 비고 |
|---|---|---|
| 자료 묶음 | `data/datasets/fly-loop-{v1,v2,v2-ln,v4-pilot,v4}` | 카드 · SHA256SUMS · `dataset verify` · v2 · v4 카드의 재배선 씨앗 표기는 틀림(파일이 정본 · 56d4b73) |
| 그래프 · 대응표 | `data/graph/traced-v1*` · `…retino-v2.shuffled-{azimuth,elevation}[-type]-s0~4.json` | 뒤섞은 대응표 20 개(눈 안 · 두 눈까지) |
| 투영기 저장본 | `runs/projector-<뇌>-train-v{4,5,6}/` | sha256 은 10장 1.3 |
| 판정 · 성적표 | `runs/caption-judge-*.json` · `runs/scorecard-*/` | v6 부터 결정 기록 |
| 측정 | `runs/odor-{control,contrast,windows}-*.json` · `runs/linear-probe-*.json` | 연구 노트 14 §9 |
| 규칙 판 | `config/*.yaml` · `config/archive/*` · `STATS_RULE(_V3)` · `TRAIN_RULE` | 11장 §3 |

## 4. 판을 바꿔 다시 할 때(개선)

1. 바꿀 설정의 판을 올린다(옛 판은 `config/archive/`) — 무엇을 다시 해야 하는지는 11장 §3.
2. 과제를 바꿨으면 `20-odor/m4-body-check.sh` → 입력 크기가 판정 문턱에 닿는지(`50-analysis/lr_budget.py`).
3. 수집은 **작게 먼저**(10 단계 스크립트가 collect-probe 를 먼저 돈다) · 판정 규칙은 **결과 전에** 코드에 적는다(stats-v3 처럼).
   묶은 뒤 **`config/datasets.yaml` 의 `current` 를 새 묶음으로** 바꾼다 — 명령마다 `--dataset` 기본값이 여기서 온다(bf48c2d · 옛 결과는 `--dataset` 을 직접).
4. 학습 판을 올리면 `bin/run_full_v6.sh` 를 복사해 새 판 순서를 만들고 `config/dashboard.yaml` train_rule · plan 을 바꾼 뒤 대시보드 재시작.
5. 끝나면 `90-wrap/finalize.sh` → 10장 대장 · R · 인수인계 · doc 세션에 문서 반영.

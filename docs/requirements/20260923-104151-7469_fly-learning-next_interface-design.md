# fly-learning-next — 인터페이스 계약 (I)

## 메타데이터

- **요구사항**: 7469 — [./20260923-104151-7469_fly-learning-next_req.md](./20260923-104151-7469_fly-learning-next_req.md)
- **작성일**: 2026-09-23 10:51:33
- **상태**: 승인
  <!-- 허용값(enum): 초안 | 검토 | 승인 — 상태=승인 전 P 작성 금지 (/DEV_PROCESS.md G2) -->
- **승인 근거**: 요구자 2026-09-23 *"승인 + 구현 중 변경 위임"* — 구현하며 바뀌는 결정은 아래 개정 이력에 남기고 진행한다. 무거운 실행은 따로 확인.
- **개정 이력**: (승인 후 변경 시 개정 행 추가 + 번복 대상 링크 필수)

| 판 | 날짜 | 무엇 | 번복 대상 |
|---|---|---|---|
| v1 | 2026-09-23 | 최초 — F-001 ~ F-005 · F-008 1~3단계는 확정 계약, F-006 · F-007 · F-008 4~5단계는 **자리만**(착수 전 이 문서 v2 로 계약) | — |
| v1.3 | 2026-09-23 | F-008 2단계(retino-v1): ① **고도만** 계산 — 기둥 좌표의 위 · 아래 축(`y = hex1 + hex2`)은 연결 지도로 확인(R7d/R8d 출력 기둥 y ≈ 64 · HSN 47 > HSE 40 > HSS 25), 눈 안의 앞 · 뒤 축은 공개 자료로 확정 못 함 → 방위는 눈 단위(좌/우) 유지 · `azimuth_deg` 칸 없음 ② **좌표를 못 얻은 뉴런(missing)은 입력에서 빼지 않고 고도 조율 없이 켠다** — LC17 이 353 중 9 개만 좌표를 얻어(다가옴 반응이 가장 센 종류) 빼면 시각 입력이 크게 약해진다 ③ 결과 4,096 개 · missing 5,105 개(시각 투사 전체) | §3.3 JSON 의 `azimuth_deg` · "missing 은 입력에서 뺀다" |
| v1.2 | 2026-09-23 | F-003 결과: ① **`sim-v2`**(`config/sim.yaml` 의 `ipsi_release` — 후각 수용 뉴런 같은 쪽 연결 × 1.4 · Gaudry 2013) — 원래 계획은 channels-v2 / task-v2 로 고치는 것이었으나 원인이 연결 세기(양쪽 반반 투사)라 뇌 판을 올렸다 ② 옛 설정 판 자리는 `config/archive/<판>.yaml`(예: `archive/sim-v1.yaml`) · `config.load(name, version=)` ③ **`stats-v2`** = v1 + 냄새 쪽 칸 판정 제외(한 창에서 입력부터 좌우가 잡음 아래) | §1 판 표 · §1 옛 판 자리 이름 |
| v1.1 | 2026-09-23 | 구현하며 바뀐 것(F-001): ① 판정 제외 채널은 `channels.yaml` 의 `judge` 칸이 아니라 **`stats.STATS_RULE["exclude_from_judgement"]`**(운동) — 채널 판을 올리지 않으려고 · 전 채널(`all`) 칸도 운동을 품어 판정 밖 ② `p_block_perm` = 진짜 대 **재배선 평균**의 결정별 차이를 에피소드로 묶은 **부호 뒤집기 순열**(에피소드 ≤ 16 이면 정확) · `p_vs_baseline` 도 기록 ③ 선형 검사 파일은 옛 칸(`brains`)을 그대로 두고 `cells` 를 더한다 ④ Holm 가족 = 판정 대상 칸 전부(보수적 — 미리 정한 가설 짝으로 좁히는 것은 결과를 본 뒤라 하지 않는다) | §1 `channels-v2` 의 `judge` · §3.2 |

> 네트워크 API · DB 는 없다. 계약 = **명령(CLI) · 파일 형식 · 판 번호 · 메타 칸**. 모듈은 core(`graph` · `sim` · 그래프 분석) · loop(`loop/*` · `datasets`).
> 기존 계약: [R-3340 I](./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md). 여기 없는 것은 그대로다.

---

## 1. 판 번호 (새로 생기는 것)

| 판 | 설정 · 코드 | 무엇이 바뀌나 | 기존 판 |
|---|---|---|---|
| `stats-v1` | `loop/stats.py` 상수 | 블록 순열 · Holm · 부트스트랩 규칙 | 없음 |
| `task-v2` | `config/task.yaml` | 물체 고도 · 지형 · 시각 입력 방식 `vision.mode: lr_sigmoid \| retinotopic` · (F-003 결과에 따라) 후각 입력 | `task-v1` 은 그대로 둔다(`fly-loop-v1` 이 쓴다) |
| `channels-v2` | `config/channels.yaml` | `space` 채널(중심복합체 · VS/HS) 추가 · (F-003) 후각 좌우가 남는 채널 · 판정 제외 표시 `judge: false`(운동) | `channels-v1` |
| `retino-v1` | `config/retino.yaml` | LC 시야 자리 계산 규칙(입력 기둥 뉴런 · 가중 · 눈 방향 변환) | 없음 |
| `compress-v1` | `config/compress.yaml` | 채널별 압축(PCA · PLS · k) · 관문 규칙 | 없음 |
| `train-v4` | `train_cli.TRAIN_RULE` | 압축 입력 · 채널당 선형 토큰 1 개 · 칸 토큰 손실 | `train-v3` |

**설정 판을 바꾸면 `config/<이름>.yaml` 을 새 판으로 고치고, 옛 판은 `config/archive/<이름>-<판>.yaml` 에 남긴다**(옛 자료 · 결과를 다시 돌릴 수 있게). 자료 · 결과 메타에는 쓰인 판이 모두 찍힌다.

---

## 2. 명령 (CLI)

| 명령 | 기능 | 하는 일 | 무게 |
|---|---|---|---|
| `graph --rewired-seed N [--rewired-seed M …]` | F-001 | (기존) 재배선 여럿 — `--rewired-seeds 0-19` 범위 입력을 더한다 | CPU · 재배선 하나 약 수 초 · 파일 약 200 MB씩 |
| `linear-probe --dataset D [--judge-only] [--permutations 1000]` | F-001 · F-003 | 자료에 있는 **모든 재배선 뇌**와 비교 · 블록 순열 p · Holm p · 앙상블 분포 · `judge: false` 채널은 판정에서 뺌 | CPU |
| `inspect laterality --types ORN_DM1,ORN_VA2` | F-003 | 감각 뉴런 → 다음 단계 연결의 같은 쪽 / 반대쪽 시냅스 비율 · 단계별(ORN → PN → 측각 · 버섯체) | CPU · 가벼움 |
| `retinotopy build` | F-008 | LC/LPLC 마다 시야 자리 계산 → `data/graph/traced-v1.retino-v1.json` | CPU · 가벼움 |
| `bump-check [--seeds …] [--dark]` | F-008 | 몸 없이 방향 회전 자극(시야 자리별 LC) → EPG 위상 · 방향 원형 상관 · 단서 끈 뒤 유지 | CPU · 뇌만 |
| `collect --name N --seeds … --rewired-seeds 0-4` | F-002 | (기존 확장) 한 에피소드에 진짜 + 재배선 여럿 · 감각 입력(자극 발화율)도 저장 | **무거움**(몸 + 뇌 여럿) |
| `collect-probe --episodes 2 --rewired-seeds 0-4` | F-002 | 작은 측정 — 결정당 시간 · 최대 메모리 · 전체 추정(저장 없음) | 가벼움 |
| `compress fit --dataset D` · `compress gate --dataset D` | F-004 | 뇌마다 압축기 맞춤 → 관문 표 | CPU |
| `train --brain real\|rewired-sK` | F-005 | (기존 확장) train-v4 · 압축 입력 | **무거움**(Qwen) |
| `eval-caption` · `scorecard` | F-001 · F-005 | (기존 확장) 재배선 투영기 여럿과 비교 · 구간 · 문구 규칙은 **앙상블 비교**로 | **무거움**(Qwen) |

"무거움"은 전부 `guard.heavy()` 를 지난다. 새 무거운 실행은 **작은 측정 명령을 먼저**(`collect-probe` · `train-probe`).

---

## 3. 파일 형식

### 3.1 학습 자료 v2 — `data/datasets/fly-loop-v2/{train,eval}/`

| 파일 | 형식 | v1 과 다른 점 |
|---|---|---|
| `features-real.npz` | 채널 → float32 [결정, 차원] | 같음 |
| `features-rewired-s<K>.npz` | 같음 · 재배선 씨앗마다 하나 | v1 은 `features-rewired.npz` 하나(= s0). **읽기는 둘 다 받는다**(`rewired` → s0) |
| `stim.npz` | `rates` float32 [결정, 구간 3, 자극 뉴런] · `stim_idx` | **새로** — 감각 입력을 남겨 나중에 다른 뇌에 다시 넣을 수 있게(재배선을 늘릴 때 몸을 다시 안 돌린다) |
| `samples.jsonl` | 결정마다 한 줄 | `truth` 에 3차원 칸: `looming_elevation_deg` · `body_roll_deg` · `body_pitch_deg` · `pos_xy` · `heading_deg` |
| `meta.json` | | `rewired_seeds` 목록 · 판(`task-v2` · `channels-v2` · `retino-v1`) |

`datasets.load(name, split, brain)` — `brain` = `real` · `rewired`(= s0) · `rewired-s<K>`.

### 3.2 선형 검사 결과 — `runs/linear-probe-<시각>.json` (stats-v1)

```json
{"dataset": "fly-loop-v2", "stats": "stats-v1", "rewired_seeds": [0,1,2,3,4],
 "cells": {"vision|loom_side": {
    "real": 0.743, "baseline": 0.667,
    "rewired": {"values": [0.59, …], "mean": 0.60, "p05": 0.55, "p95": 0.66},
    "rank_pct": 100.0, "p_block_perm": 0.004, "p_holm": 0.03,
    "ci95_real": [0.66, 0.82], "judge": true, "pass": true}}}
```

`pass` = `real` 이 앙상블 95 % 위(`rank_pct` ≥ 95) **그리고** `p_holm` < 0.05 **그리고** `real` > `baseline`(R-7469 F-001 정의).
블록 = 에피소드(씨앗). 순열 = 에피소드 단위로 라벨을 섞는다.

### 3.3 LC 시야 대응 — `data/graph/traced-v1.retino-v1.json`

```json
{"rule": "retino-v1", "source": "assignedOlHex1/2 (공식 주석) · 입력 시냅스 가중 평균",
 "neurons": [{"idx": 123, "bodyId": 1000, "type": "LC4", "side": "R",
              "hex": [12.3, 20.1], "azimuth_deg": -35.0, "elevation_deg": 10.0,
              "input_synapses": 812, "input_columns": 37}],
 "missing": {"LC21": 3}}
```

기둥 좌표 → 눈 방향 변환은 `config/retino.yaml` 의 규칙(출처와 함께). 좌표를 못 얻은 뉴런은 `missing` 에 세고 입력에서 뺀다(채우지 않는다).

### 3.4 압축기 · 관문 — `runs/compress-<dataset>-compress-v1/`

| 파일 | 내용 |
|---|---|
| `<brain>.npz` | 채널마다 `mean` · `components` [k, d] · (PLS 면 `weights`) |
| `gate.json` | 채널 × 칸 관문 결과(3.2 의 `pass` 기준 · 압축 뒤 값으로) · 통과 채널 목록 |
| `meta.json` | 규칙 · 고른 k · 설명 분산 · 자료 체크섬 |

압축기는 **학습 나눔(train)으로만** 맞춘다 · 뇌마다 따로(재배선도 같은 규칙).

### 3.5 투영기 train-v4 — `runs/projector-<brain>-train-v4/`

`meta.json` 에 더해지는 칸: `compress`(폴더 · 판 · 통과 채널) · `loss_mask`(칸 토큰 규칙) · `slots`(통과 채널 수 × 1).
입력 글의 자리 표시는 **통과 채널만** `<채널>▢</채널>`(채널당 1 개). 손실은 캡션 칸 값 · 행동 JSON 값 토큰에만(틀 · 키 이름 · 구두점은 가림).

---

## 4. 호출 순서 (권하는 순서 · G1)

```
F-001  linear-probe(stats-v1) ── fly-loop-v1 로 먼저(재배선 1 개 · 순열 · Holm)
F-003  inspect laterality → (고칠 판 결정) → channels-v2 / task-v2
F-008  retinotopy build → task-v2(고도 · 지형 · retinotopic) → channels-v2(space) → bump-check
F-002  collect-probe → [요구자 확인] → collect --rewired-seeds 0-4 → dataset pack fly-loop-v2
F-001  linear-probe fly-loop-v2(재배선 5 · 필요하면 stim.npz 로 재배선 추가)
F-004  compress fit → compress gate
F-005  train-probe → [요구자 확인] → train real · rewired-s0..s2 → eval-caption · scorecard(씨앗 40)
(F-006 · F-007 · F-008 4~5 — 이 문서 v2 에서)
```

---

## 5. 자리만 잡아 둔 계약 (v2 에서 확정)

| 기능 | 정할 것 |
|---|---|
| F-006 | 학습 매개변수 목록(종류별 시간 상수 · 문턱 · 유형 쌍 배율)과 판 `sim-v2` 계열 · 기울기 방식(대리 기울기 / 속도 근사) · 학습 목표 파일 · 연결 가중치 불변 검사 |
| F-007 | 가소성 규칙 판 · KC→MBON 연결 목록 · 보상 신호 · 대조군(끔 · 무작위 보상 · 재배선) 명령 |
| F-008 4~5 | 공간 목표(봉우리 · 방향 · 위치 · 고도) 학습 규칙 · 3차원 캡션 서식(`caption-v2`) |

---

## 6. 호환성

- `fly-loop-v1` · `projector-*-train-v2` · `train-v3` 명령은 그대로 돈다(`--version` · `rewired` = s0).
- 옛 설정 판은 `config/archive/` 에서 불러 쓸 수 있다(`config.load(name, version=…)` 를 더한다).

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
| v2.1 | 2026-09-23 | 자료 v2 결과로 바뀐 것: ① **`compress-v2`** = 모든 뇌 · 채널에 같은 차원 64(v1 의 분산 0.95 규칙은 진짜 시각 16 · 재배선 64 로 절차가 달라졌고 시각 정보가 잘렸다 — 결과 뒤 고침 · 까닭은 `config/compress.yaml` 주석) · train-v4 의 `compress` 도 v2 ② `compress.out_dir/load(version=)` — 학습 규칙이 정한 판을 읽는다 ③ 봉우리 검사 결과: 진짜 배선 EPG 발화 0/3,297 — **봉우리 없음**(F-008 3단계 실패로 기록 · F-006 space 목표에 그대로) | §3.4 · §1 `compress-v1` |
| **v2** | 2026-09-23 | **F-006 · F-007 · F-008 4~5단계 계약**(§7 ~ §9 · 재료 = [연구 노트 13](../research/fly-connectome/13-회로-학습-설계-조사.md)) + **retino-v2**(앞 · 뒤 축 확정 → 방위) · 새 판 `sim-v3` · `physiology-v1` · `plasticity-v1` · `task-v3` · `retino-v2` · `fit-probe` · `fit` · `plasticity-probe` 명령. 착수는 F-004 · F-005 판정 뒤 · 무거운 실행은 요구자 확인 | §5 "자리만" · §1 판 표 |
| v1.5 | 2026-09-23 | F-005 코드: ① 손실은 틀 토큰을 **0 이 아니라 0.1 배**(`template_weight`) — 시스템 글에 캡션 서식이 없어 서식을 배울 곳이 정답뿐이다 · 칸 값 = 캡션 `: 값 /` · 행동 JSON `:값` (토크나이저 글자 위치로 가림) ② 투영기 `kind: linear` · 채널당 토큰 1 · 입력 = 관문 통과 채널의 압축 벡터 · 관문 통과 0 이면 학습하지 않는다 ③ `--brain rewired-s<K>` · 학습 · 평가 기본 자료 `fly-loop-v2` ④ eval-caption · scorecard 가 meta 의 압축을 똑같이 적용 · 성적표의 재배선 앙상블(여러 투영기)은 학습 뒤 추가 | §3.5 "틀 · 키 · 구두점은 가림" |
| v1.4 | 2026-09-23 | ① `bump-check` 명령 → **`inspect bump --dataset`**(자료 v2 의 공간 채널과 참값 몸 방향으로 · 뇌만 돌리는 방향 회전 검사는 하지 않는다 — 시각 입구가 다가옴뿐이라 방향 단서가 없다) ② `linear_probe.run(transform=)` — 압축 관문이 같은 판정 코드를 쓴다 ③ 수집 조각 이름 `v2main` · `v2eval`(v1 조각 보호) | §2 `bump-check` |
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

## 5. (v1 의 "자리만" — v2 에서 §7 ~ §9 로 확정)

## 6. 호환성

- `fly-loop-v1` · `projector-*-train-v2` · `train-v3` 명령은 그대로 돈다(`--version` · `rewired` = s0).
- 옛 설정 판은 `config/archive/` 에서 불러 쓸 수 있다(`config.load(name, version=…)` 를 더한다).

---

## 7. F-006 — 세포 종류 단위 회로 학습 (v2 · 근거 연구 노트 13 §2)

### 7.1 무엇을 배우나 — 판 `sim-v3`

| 매개변수 | 단위 | 초기값(sim-v2) | 범위(선언 · 로그 매개화) | 불변 |
|---|---|---|---|---|
| `tau_ms[t]` 막 시간 상수 | 종류 t | 20 | 5 ~ 200 ms | — |
| `v_th_mV[t]` 문턱 | 종류 t | −45(쉬는 값 위 7 mV) | 쉬는 값 위 2 ~ 15 mV | 이득은 두지 않는다(문턱과 퇴화) |
| `alpha[p]` 종류 쌍 배율 | 쌍 p(t → u) | 1 | 0.1 ~ 10 · **α ≥ 0** | 연결 수 · 부호 · 상대 · `w_scale` 0.25 · `ipsi_release` |
| (선택) `b_mV[t]` 상수 입력 | 종류 t | 0 | 0 ~ 5 mV · **출처 있는 종류만** | |

**범위(scope) 이름** — 학습 종류 집합과 쌍 집합을 이름으로 잠근다(연구 노트 13 §2.1 실측):

| scope | 종류 | 쌍(≥ 100 시냅스) | 매개변수 수(추정) |
|---|---|---|---|
| `escape-small` | 다가옴 LC/LPLC 13 종 + DNp01 · DNp02 · DNp04 · DNp11 | 클래스 쌍 몇 개 | **≤ 50** — 1단계 |
| `space` | 중심복합체 21 종 | 83 | 125 |
| `escape` | 시각 투사 346 + 하강 480 | 473 | 2,125 |
| (하지 않음) 채널 내부 ≈ 13,450 · 뇌 전체 ≈ 18 만 | | | 자료보다 많다 |

매개변수 파일 `data/graph/traced-v1.params-<판>-<scope>.npz`: `types[T]` · `tau_ms[T]` · `v_th_mV[T]` · `alpha[P]` · `pair_id[E]`(int32 · 쌍 밖은 −1) · `scope` · `seed` · `targets` 판.
**커넥톰 정체성 검사**(시험 · 학습 뒤 필수): `edges.npz` 해시 · 부호 배열 · `alpha ≥ 0` · 매개변수 파일 밖에 바뀐 것 0.

### 7.2 어떻게 배우나 — 두 단계 (순서 고정)

| 단계 | 방법 | 명령 | 메모리 · 시간(추정 — `fit-probe` 로 먼저 잰다) | 요구자 확인 |
|---|---|---|---|---|
| 1 | **기울기 없음 CMA-ES** — 정방향만 · λ = 4 + ⌊3 ln P⌋ · 손실 = 생리 목표(`physiology-v1`) | `fit --scope escape-small --method cmaes --targets physiology-v1 --brain real\|rewired-s<K>` | 정방향 상태만(3.3 MB × λ) · 시간 = 세대 × λ × 창 → **P ≤ 50 · 창 20 · 세대 100 부터** | **예** |
| 2 | **희소 대리 기울기 + 잘림 BPTT + √T 체크포인트** — 함수형 걸음(지금 엔진과 같은 순서 · 제자리 없음) · 기울기는 `grad_scope` 뉴런(N_g)에만 · 나머지 detach | `fit --scope space\|escape --method sgbptt --targets physiology-v1 --brain …` | BPTT 저장 ∝ B × T × N_g × 14 B(뇌 전체 3,000 걸음이면 6.9 GB · autograd ×2~4 → 14~28 GB → **체크포인트 필수** ≈ 0.3 GB) · 창당 4~6 s → 자료 v2 3,000 결정 ≈ 4 h/뇌 | **예** |
| (막히면) | 온라인 흔적(e-prop 류 · T 무관) → 속도 근사(이식 검증 뒤에만) | — | — | — |

- `fit-probe --scope … --window-ms 10,30 [--method …]`: 창 두 길이로 최대 메모리 · 시간을 재고 **무엇에 비례하는지** 기록 → 요구자 확인 자료. 저장 없음.
- **금지**: 걸음마다 연결 전체(E)를 곱하는 방식(T × E = 300 GB · ConnecTorch 식). 도착한 연결만 쓴다.
- 시험: 대리 기울기를 끈 함수형 정방향이 지금 엔진과 **발화 완전 일치**(작은 그래프 · CPU).

### 7.3 무엇을 목표로 — 판 `physiology-v1` (`config/physiology.yaml` · 결과 보기 전에 잠근다)

손실 L = Σ_k w_k · (측정_k − 목표_k)² / 허용폭_k² + L_과제(관문 통과 칸의 선형 해독) + λ · ‖매개변수 − 초기값‖².
항목(연구 노트 13 §2.4 표에서 · 각 항목에 출처 · 허용폭 · 확인 수준 [P]/[A]/[S]):

| 회로 | 목표 | 출처 |
|---|---|---|
| 도망 | GF 반응 0.5~1 · 반응당 1~2 · 잠복 10~60 ms · 뇌 전체 < 0.05 Hz · 재배선 < 0.1(= `bench/escape.py` 5 항목) | von Reyn 2014/2017 |
| 다가옴 조율 | 13 사구체 상대 세기(`loom.yaml`) · LPLC2 팽창 선택 | Turner 2022 · Klapoetke 2017 |
| 나침반(space) | EPG 봉우리 FWHM 82.3 ± 11.5° · 봉우리 속도 = 몸 각속도 | Seelig & Jayaraman 2015 · Turner-Evans 2017 |
| 수평 · 수직계 | HSN/HSE/HSS 수용장 순서(y 축 확인에 쓴 것) | Schnell 2010 |
| 후각 · KC | 냄새당 KC 5~10 % 만 발화 | Lin 2014 |

퇴화 경고(Beiran & Litwin-Kumar 2025): 과제 손실만으로는 해가 여럿 → 생리 항 필수 + **씨앗 앙상블 5~10** 의 퍼짐을 결과에 적는다.

### 7.4 대조군 · 성공 기준 · 결과물

- 재배선 s0~s4 에 **같은 scope · 같은 매개변수 수 · 같은 절차 · 같은 씨앗**. 학습 뒤 진짜 > 재배선 앙상블(stats-v2) · 도망 5 항목 유지 · **재배선도 똑같이 오르면 실패로 기록** · 학습 전 대 후(진짜).
- 결과 `runs/fit-<scope>-<판>-<brain>-<시각>/`: `params.npz` · `loss.jsonl`(세대/걸음마다) · `meta.json`(scope · method · targets 판 · 씨앗 · 정체성 검사 결과 · 메모리 최대) · 도망 재채점 기록.
- 학습 안내서 10장 대장에 한 줄(재배선과 나란히).

---

## 8. F-007 — 버섯체 가소성 (v2 · 근거 연구 노트 13 §3)

### 8.1 그래프 실측(2026-09-23 · 조회만)

KC 4,064(전부 ACh) · MBON 97(37 종류) · DAN 332(PAM 316 · PPL1 16 · 부호 0 그대로) · **KC → MBON 61,210 연결 · 463,640 시냅스**(연결당 중앙 5) · DAN → KC 127,599 · PN → KC 22,586(KC 당 PN 5.6).
구획 = "MBON 에 시냅스가 가장 많은 DAN 종류"로 자료에서 정한다 — 상위 쌍이 문헌 명명법과 전부 일치(PPL101 → MBON11 γ1pedc · PAM01 → MBON01 γ5 · PAM08 → MBON05 γ4 · PAM11 → MBON07 α1 · PPL103 → MBON12 γ2α′1 · PPL106 → MBON14 α3 …).

### 8.2 규칙 — 판 `plasticity-v1` (`config/plasticity.yaml` · 값은 선언 · 창의 경계만 원문)

| 부품 | 계약 |
|---|---|
| 바뀌는 곳 | KC → MBON 61,210 연결의 **배율 m_ij ∈ [m_min 0.1, 1]**(초기 1)만. 존재 · 시냅스 수 · 부호 불변 · DAN 부호 0 그대로(조절 신호로만 읽음) |
| 3요소 규칙 | KC 흔적 ē_i(τ_e **5 s** · 1~20 선언) · 구획 신호 D_c(τ_D **0.2~0.5 s**) · **dm_ij/dt = −η · D_c(j) · ē_i · m_ij**(곱셈형 · 앞 짝짓기 약화만 · 뒤 짝짓기 강화는 2판) · 에피소드 안 회복 없음 |
| η | 값이 아니라 **목표로 보정**: 짝짓기 한 번(냄새 1 s × DAN 1 s)에 그 구획 MBON 의 CS+ 구동 50~90 % 감소 · 짝 안 지은 냄새 < 25 %(Hige 2015) — `plasticity-probe` 가 잰다 |
| 가르치는 신호 | **세계 사건 → DAN 직접 자극**(포아송 킥 · 약 1 s · Aso & Rubin 2016 광유전 설계 흉내 · 회로가 스스로 DAN 을 못 켬): 먹이 도착 → 보상 DAN PAM01/15 · PAM02 · PAM07/08 · PAM03/04 · PAM11 → MBON01/02/03/05/07(회피)이 약해짐 · 부딪힘 → 처벌 DAN PPL101/103/106 → MBON11/12/14(접근)이 약해짐 |
| 파일 | `runs/plasticity-<판>-<brain>-<시각>/m.npz`(연결 배율 · 61,210) · `meta.json`(규칙 · η · 구획 표 · 사건 기록) |

### 8.3 과제 · 읽기 · 대조군 — 판 `task-v3`

- `task-v3`: 냄새 **둘** — CS+ = 지금 먹이 냄새(ORN_DM1 · VA2) · CS− = 다른 사구체 ORN 집합(선언 · 출처) · 먹이는 CS+ 에서만 · 학습 에피소드 N → 시험 에피소드(보상 없음).
- 1차 기준 = `value` 채널(MBON)의 CS+/CS− **선형 해독**(stats-v2 · 학습 전 대 후). 2차 = 회로가 낸 하강 뉴런(DNa02 좌우) — **MBON → 하강 경로가 침묵 LIF 에서 발화를 내는지 먼저 잰다**(못 내면 "못 잼"으로 기록).
- **대조군 다섯**: ① 가소성 끔 ② 무작위 보상(같은 수의 DAN 자극을 사건과 무관하게) ③ 재배선 — **PN → KC 를 섞는다**(KC → MBON 만 섞는 것은 약함) ④ DAN ↔ 구획 바꿔치기(PAM ↔ PPL1 · 정서가 뒤집혀야 규칙이 구획을 쓰는 증거) ⑤ 학습 전.
- 성공 = 학습 뒤 CS+/CS− 해독이 찍기를 넘고 ①②③ 모두보다 높다(stats-v2) · ④ 에서 정서가 뒤집힌다.
- **먼저 잴 것**(`plasticity-probe` · 몸 없이 뇌 창 몇 개 · 가벼움 · 잠금 뒤): (i) CS+ · CS− 자극 시 KC 발화 비율(문헌 5~10 %) (ii) MBON 이 KC 입력으로 발화하나 (iii) DAN 자극이 구획 MBON 을 바꾸나 (iv) η 보정. 0 % 나 50 % 면 손으로 고치지 않고 기록(F-006 목표로).

---

## 9. F-008 4~5단계 · retino-v2 (v2 · 근거 연구 노트 13 §4)

### 9.1 앞 · 뒤 축 — 확정 (2026-09-23)

두 근거가 독립적으로 일치한다: ① 공개 정의(Nern 연구실 Cell 2026 방법: `h = hex1 − hex2` · "high h = anterior" · Nern 2025 그림 3a) ② 우리 그래프에서 LC4 → DNp02 시냅스가 x(= hex2 − hex1)와 −0.76/−0.85(L/R), LC4 → DNp11 이 +0.73/+0.80 — Dombrovski 2023 의 "앞쪽 LC4 → DNp02 · 뒤쪽 → DNp11" 과 방향이 같다(`runs/lc4-dn-axis-20260923.json`). GF 에 기울기 없음 · 등쪽 LPLC2 → GF 가 셈(y +0.68/+0.77)도 논문과 같다.
→ **앞쪽 = hex1 − hex2 큰 쪽 · 등쪽 = hex1 + hex2 큰 쪽.**

### 9.2 `retino-v2` (`config/retino.yaml` 판 올림 · `retinotopy build` 그대로)

| 항목 | 계약 |
|---|---|
| 좌표 이름 | `h = hex1 − hex2`(앞 +) · `v = hex1 + hex2`(등 +) — 공개 정의와 맞춘다 |
| 방위 | h 의 [최소, 최대] → `azimuth_range_deg`(선언 · 앞쪽 0° 근처 양안 겹침 허용) 선형. 왼쪽 눈은 거울(부호 반전) |
| 고도 | v1 과 같음 |
| JSON | `azimuth_deg` 칸 추가(v1 의 "확정 못 함" 문구 제거) · `missing` 규칙 같음(빼지 않고 조율 없이) |
| 인코더 | `task-v3` 의 `vision.mode: retinotopic2` — 좌우 시그모이드 대신 **LC 마다 (방위, 고도) 가우시안**(σ 방위 · 고도 선언) |
| 대조군 | 뒤섞은 대응표 **둘**(방위 · 고도) |
| 보조 검사(선택) | T4a/T4b 수상돌기 벡터로 h 부호 재확인(T4c/d 로 v 먼저) |

⚠️ 자료 v2 는 retino-v1(고도만)로 모았다 → retino-v2 는 (a) `stim.npz` 대신 참값(`samples.jsonl` 의 물체 방위 · 고도 · 크기 · 확대 · 냄새 · 다리)으로 감각 입력을 **다시 계산해 뇌만 다시 돌리는 재주입**(`collect --replay fly-loop-v2`) 또는 (b) v3 자료에 쓴다. 재주입은 뇌 6 × 3,400 결정 ≈ 몸 없이 3 h(추정 · 먼저 잰다).

### 9.3 F-008 4단계(공간 학습) · 5단계(3차원 캡션)

- 4단계 = F-006 의 `scope space` + `physiology-v1` 의 나침반 항목 + 과제 항(방향 · 위치 · 고도 해독). 봉우리 검사(`inspect bump`)가 실패로 나오면 그 사실을 목표에 그대로 둔다(억지로 맞추지 않는다).
- 5단계 = `caption-v2` 서식: 기존 다섯 칸 + `방향: N°` · `먹이: 거리 mm · 방위` · `물체 고도: 위/정면/아래` · `몸 기울기: …`. 투영기에는 관문을 통과한 저차원 채널만. 대조군: 재배선 앙상블 · 채널 뒤섞은 입력 · 회로 없음.

---

## 10. 판 번호 (v2 에서 더해진 것)

| 판 | 자리 | 무엇 |
|---|---|---|
| `sim-v3` | `config/sim.yaml` + 매개변수 파일 | 종류별 τ · 문턱 · 쌍 배율(scope) — 학습 결과 |
| `physiology-v1` | `config/physiology.yaml` | F-006 손실의 생리 목표 목록(출처 · 허용폭) |
| `plasticity-v1` | `config/plasticity.yaml` | KC → MBON 3요소 규칙 · 구획 표 · DAN 사건 대응 |
| `task-v3` | `config/task.yaml` | 냄새 둘(CS+/CS−) · `vision.mode: retinotopic2` |
| `retino-v2` | `config/retino.yaml` | h · v · 방위 |
| `caption-v2` | `loop/teacher.py` | 3차원 칸 서식 |

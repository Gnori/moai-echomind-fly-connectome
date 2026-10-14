# fly-learning-next — loop 검증 보고 (W)

## 메타데이터

- **요구사항**: 7469 — [./20260923-104151-7469_fly-learning-next_req.md](./20260923-104151-7469_fly-learning-next_req.md)
- **대응 P**: [./20260923-104151-7469_fly-learning-next-loop_dev-plan.md](./20260923-104151-7469_fly-learning-next-loop_dev-plan.md)
- **관련 I**: [./20260923-104151-7469_fly-learning-next_interface-design.md](./20260923-104151-7469_fly-learning-next_interface-design.md)
- **작성일**: 2026-09-26
- **범위**: loop P 의 Task 1~5 = F-001(판정 기준) · F-008 1~3단계의 loop 쪽 · F-002(자료 v2) · F-004(압축 · 관문) · F-005(작은 투영기 · train-v5)
- **결과**: **조건부 통과** — 코드 · 실행 · 판정은 P 대로 끝났다.
  **캡션(Qwen 이 신호를 말로 옮기기)은 진짜 배선이 재배선 셋을 모두 이겼다**(물체 있음 · 다리 접촉).
  그러나 **성적표(먹이 찾아가기)는 이기지 못했다** — vector 0.075 = 재배선 0.075 · 0.025 · 0.075 · p 0.42 →
  **R 2.0-6 문구 규칙 두 번째 발동**: 행동 결과에 "초파리 뇌"라는 말을 쓰지 않는다.

> 원칙: 서술은 증거가 아니다 — 아래 로그 · 파일 · sha256 만 증거다. 파일은 `runs/`(git 밖 · 백업 두 곳 대조 일치).

---

## 1. 변경 파일 목록

| 파일 | 변경 유형 | 요약 |
|---|---|---|
| `src/moai_echomind_fly/loop/stats.py` · `linear_probe.py` | 추가 · 수정 | stats-v1 → v2(에피소드 블록 순열 · Holm · 부트스트랩 · 판정 제외 채널) · 핵 고유분해로 모든 λ · 칸 한 번에 |
| `src/moai_echomind_fly/loop/{world,encoder,signals,episode}.py` · `body/` | 수정 | task-v2(물체 고도 · 3차원 참값 칸) · 인코더 `retinotopic`(LC 고도 가우시안) · channels-v2(`space`) |
| `src/moai_echomind_fly/loop/collect.py` · `datasets.py` | 수정 | `--rewired-seeds` · `stim.npz` · `collect-probe` · 자료 `fly-loop-v2` |
| `src/moai_echomind_fly/loop/compress.py` · `config/compress.yaml` | 추가 | compress-v1 → v2(모든 뇌 64 차원) → **v3(채널마다 평균 크기 1)** · 선형 관문 |
| `src/moai_echomind_fly/loop/{projector,train,train_cli}.py` | 수정 | train-v4 → **train-v5**(압축 입력 · 채널당 토큰 1 · 칸 가중 1 / 틀 0.1 · `--train-seeds`) |
| `src/moai_echomind_fly/loop/caption_score.py` | 추가 | 채점 평가 eval-v2 → **v2.1**(빈칸 채우기 · 정답 접두 · 쪽은 있다고 준 채 세 방향) · `judge-captions`(판정 칸 넷 · stats-v2) |
| `src/moai_echomind_fly/loop/scorecard.py` | 수정 | scorecard-v2(재배선 앙상블 · 씨앗별 부호 뒤집기) |
| `bin/run_full_v4.sh` · `bin/run_full_v5.sh` | 추가 | 전체 순서 · 끝 코드 바로 받기 · Qwen 단계 여유 63 % 대기 |
| `config/{task,channels}.yaml` · `config/archive/*` | 수정 | task-v2 · channels-v2 · 옛 판 보관 |
| `tests/{test_stats,test_caption_score,test_loop_parts,test_train_layerwise,…}.py` | 추가 · 수정 | 아래 2장 |

커밋(loop 쪽 주요): `bb740c2`(stats · 판정) · `14eef3b` · `41017e5`(task-v2 · retinotopic · channels-v2) · `a2b8c94`(compress-v1) · `804af0c`(train-v4) · `80c18d4`(scorecard-v2) ·
`5765b43`(선형 검사 고유분해) · `7304375`(compress-v2) · `ba116ab`(`--train-seeds`) · `14692eb`(끝 코드) · **`465c7f0`(compress-v3 · train-v5)** ·
`61a5350` · `113ebde` · `0147f58`(채점 평가) · `081560c`(judge-captions) · `152cfbe` · `0c80369`(평가 대기 문턱) · `9c1f053`(성적표 기록 · 10장 대장).

---

## 2. 테스트 실행 로그 (원문 · 2026-09-26)

```
$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q -m "not heavy"
94 passed, 3 deselected, 18 warnings in 12.83s
```

칸 가림(F-005 인수 기준 "틀 토큰에 기울기 0 · 칸 토큰에만")은 두 시험이 맡는다:
`tests/test_stats.py::test_value_spans_and_token_weights_mark_only_field_values`(칸 값 토큰만 표시) ·
`tests/test_train_layerwise.py::test_weighted_loss_equals_mean_when_uniform_and_ignores_zero_weights`(가중 0 토큰은 손실에 안 들어감).
⚠️ train-v4/v5 는 틀 토큰을 **0 이 아니라 0.1** 로 둔다(`804af0c` · `loop/train_cli.py` 주석: 시스템 글에 캡션 서식이 없어 서식을 배울 곳이 정답뿐) — 인수 기준 문구와 다르게 구현했음을 적는다.

---

## 3. 모듈 동작 확인 기록

### 3.1 F-001 판정 기준 (Task 1)

| 실행 | 파일 | 결과 |
|---|---|---|
| 자료 v1 선형 검사(재배선 1) | `runs/linear-probe-20260923-105625.json` | 통과 2 칸(시각 → 물체 거리 · 움직임 · Holm 0.047) |
| **자료 v2 선형 검사(재배선 5 · 40 에피소드 376 결정)** | `runs/linear-probe-20260923-201422.json`(sha256 `4954196c49af9a05`) | **통과 8 칸** — 시각 → 물체 있음 0.851 · 쪽 0.801 · 거리 0.771 · 움직임 0.830(재배선 0.62~0.73 · 찍기 0.726 · Holm 0.004) · 다리 ← 몸 0.622 · 촉각 0.569 · 구애 0.559 · 미각 0.513(재배선 ~0.46 · 찍기 0.362) |

### 3.2 F-008 1~3단계 loop 쪽 (Task 2)

- task-v2 · retinotopic 인코더 · channels-v2 — 시험 `test_retinotopic_encoder_prefers_lc_at_object_elevation` 통과.
- 봉우리 검사 `runs/bump-check-20260923-195413.json`: **진짜 배선 EPG 발화 0 / 3,297 결정 — 봉우리 없음(실패 · 그대로 기록)** · 재배선 s4 만 793 결정에 무작위 발화 → F-006 `space` 목표로 넘김.
- 지형은 미룸(FlyGym 1.x OdorArena 는 평지) · 뒤섞은 대응표 대조와 앞뒤 축(retino-v2)은 core Task 7 · 자료 재주입 때.

### 3.3 F-002 자료 v2 (Task 3)

- 작은 측정 `runs/collect-probe-20260923-111821.json`: 뇌 6 · 결정당 12.0 s · 일꾼 3.4~3.6 GB → 요구자 "지금 시작".
- 수집 2026-09-23 11:21 → 19:33(끝 코드 0 · 0) → `data/datasets/fly-loop-v2`(학습 2,921 결정 / 300 에피소드 · 평가 376 / 40 · 재배선 s0~s4 · 자료 카드에 `retino-v1`).

### 3.4 F-004 압축 · 관문 (Task 4)

| 판 | 무엇 | 관문 결과 |
|---|---|---|
| compress-v1 | 분산 0.95 규칙 | 진짜 16 차원 · 재배선 64 → 절차가 달라 시각이 잘림 · 통과 몸 · 미각뿐 → 폐기 |
| compress-v2 | 모든 뇌 64 차원 | 통과 시각 · 촉각 · 미각 · 몸(칸 7) — 그러나 **입력 크기가 뇌마다 다름**(시각 진짜 0.90 대 재배선 1.77~2.07) |
| **compress-v3** | 채널마다 학습 평균 L2 로 나눠 크기 1 | 통과 **시각 · 촉각 · 미각 · 몸**(칸 7 · v2 와 같음) · 모든 뇌 · 채널 평균 크기 1.000 실측 · `runs/compress-fly-loop-v2-compress-v3/gate.json`(sha256 `7202920b61a77d98`) |

### 3.5 F-005 작은 투영기 · train-v5 (Task 5)

**v4 는 비교에 쓰지 않는다** — 재배선 s0 가 캡션 450 걸음에서 진짜보다 훨씬 빨리 배움(3.42 대 2.12) → 원인 = compress-v2 의 입력 크기 불공정 → 요구자 결정 A 로 중단(`runs/projector-rewired-s0-train-v4.stopped-1042/`).

train-v5 학습(`bin/run_full_v5.sh --skip-compress` · 2026-09-24 12:43 → 09-25 14:45 · 뇌마다 3,923 걸음 · 최대 31.4 GB 평탄 · **Qwen 지문 네 뇌 모두 `a4cfcb8d0ae14b11`** — 투영기만 배움):

| 뇌 | 걸린 시간 | 최종 검증 캡션 | 최종 검증 행동 | `stage1-caption` sha256 | `final` sha256 |
|---|---|---|---|---|---|
| 진짜 | 8 h 25 m | **1.754** | 0.57 | `22200ac8d7502ae1` | `9e0c21c2e2fac9a6` |
| 재배선 s0 | 5 h 55 m | 1.873 | 0.575 | `a645e551d8f1238b` | `e0babfee3b1a958a` |
| 재배선 s1 | 4 h 58 m | 1.911 | 0.597 | `46ae000cfb9d6ef1` | `f0aa1610738020b8` |
| 재배선 s2 | 약 6 h 44 m | 1.768 | — | `75c763229638af39` | `07498e0007209dd3` |

(손실은 판정이 아니다 — 판정은 아래 채점 평가와 성적표.)

**캡션 판정** `runs/caption-judge-train-v5-final.json`(sha256 `942d414ba3eb0c9f` · eval-v2.1 · stats-v2 · 판정 칸 넷은 평가 전에 정함 `081560c` · 40 에피소드 376 결정):

| 칸 | 진짜 | 재배선 s0 · s1 · s2 | 찍기 | Holm p | 판정 |
|---|---|---|---|---|---|
| 물체 있음 | **0.806** | 0.729 · 0.729 · 0.739 | 0.726 | 0.0004 | ✅ 통과 |
| 물체 쪽(23 에피소드 103 결정) | 0.573 | 0.466 · 0.340 · 0.544 | 0.505 | 0.026 | 규칙상 통과 · ⚠️ 찍기 대비 p 0.14 → **찍기와 구별 안 됨** |
| 물체 움직임 | 0.524 | 0.524 · 0.524 · 0.563 | 0.524 | 1.0 | ❌ 불통 |
| 다리 접촉 | **0.574** | 0.505 · 0.497 · 0.391 | 0.362 | < 0.001 | ✅ 통과 |

캡션 단계본(`runs/caption-judge-train-v5-stage1-caption.json` · sha256 `a942eb67dde56d2d`): 물체 있음 **0.851**(재배선 0.721~0.745) · 쪽 0.621 · 다리 **0.582** — **최종본보다 높다**(행동 단계가 캡션을 조금 깎는다 · R-3340 W 잔여 3 과 같은 방향).

**성적표** `runs/scorecard-20260925-175025/scorecard.json`(sha256 `4bbd8625b19836b1` · scorecard-v2 · 씨앗 2000~2039 × 7 구성 = 280 에피소드 · 09-25 17:50 → 09-26 06:57):

| 구성 | 먹이 도착률 | 95 % 구간 | 끝 거리 평균(mm) | 결정 수 |
|---|---|---|---|---|
| vector(진짜 배선) | 0.075 | 0.0 ~ 0.175 | 47.5 | 765 |
| rewired-s0 | 0.075 | 0.0 ~ 0.175 | 50.7 | 767 |
| rewired-s1 | 0.025 | 0.0 ~ 0.075 | 53.3 | 789 |
| rewired-s2 | 0.075 | 0.0 ~ 0.175 | 58.2 | 771 |
| text | 0.075 | — | 43.2 | 767 |
| **no_circuit** | **0.95** | — | 2.0 | 445 |
| no_llm | 0.0 | — | 25.2 | 800 |

판정: 순위 33 % · 부호 뒤집기 p 0.42 · `vector_beats_rewired: false` → **"섞은 배선 대조를 이기지 못했다 — '초파리 뇌'라는 말을 쓰지 않는다"**(파일의 `wording` 그대로).
해석(확정 결론 · 인수인계 C): Qwen 은 과제를 할 줄 안다(no_circuit 0.95) · 먹이 도착에 필요한 **냄새 좌우**가 어느 뇌 신호에도 없다(F-003 · 냄새 쪽 칸은 판정 제외) → 다음 병목.

### 3.6 인수 기준 대조 (R-7469 §2.2 · loop 쪽)

| 기능 | 성공 기준 | 결과 |
|---|---|---|
| F-001 | 재배선 앙상블 · 블록 순열 · Holm · 구간 | ✅ 충족 |
| F-002 | 300 / 40 에피소드 · 재배선 여러 개 같은 입력 · 작은 측정 먼저 | ✅ 충족 |
| F-004 | 규칙 먼저 · 관문 통과 칸만 다음으로 | ✅ 충족(v1 · v2 는 결과 뒤 고침 · 까닭 R 에 기록) |
| F-005 | 캡션 칸이 찍기와 재배선을 앞섬 **그리고** 성적표가 재배선보다 높음 | ⚠️ **캡션만 충족**(물체 있음 · 다리) · 성적표 불충족 → R 2.0-6 |
| F-008 1~3 (loop) | 3차원 입력 · 공간 채널 · 봉우리 검사 | ✅ 구현 · 봉우리는 **실패로 기록**(예상된 한계) |

---

## 4. 정본 문서 동기화 내역

| 문서 | 내용 | 커밋 |
|---|---|---|
| 학습 안내서 10장 대장 | v4 · v5 손실 · sha256 · 판정 · 성적표 | `b7e219e` · `9c1f053` |
| 학습 안내서 README · 09장 8절 · 연구 노트 11 §8.7 · 뿌리 README | v5 결과 · R 2.0-6 두 번째 · 다음 병목 냄새 좌우(doc 세션) | `0800205` |
| 연구 노트 11 · 12 · 학습 안내서 · `docs/datasets.md` | R-7469 판 동기화 · 자료 v2 결과 | `22b1ccf` · `07f3642` |
| 인수인계 | 요약 B · C · D | `d7b2c80` 뒤 |

---

## 5. 잔여 이슈

1. **성적표를 못 이김 — 냄새 좌우** — 먹이 찾기의 필수 정보가 입력 단계부터 잡음 아래(F-003). 다음 판 설계는 연구 노트 14(research 세션 조사 중) → 새 R 또는 F-003 보강. (인수인계 D 2)
2. **평가 자료가 얇다** — 40 에피소드 · 물체 나오는 에피소드 23 · 소리 · 암컷 · 냄새 세기는 값 하나. 물체 쪽 같은 작은 차이를 못 가림. (D 3)
3. **행동 단계가 캡션을 깎는다** — 캡션 단계본 > 최종본. (D 4)
4. **틀 토큰 가중 0.1** — 인수 기준 "기울기 0" 과 다름(의도된 구현 변경 · 2장).
5. ~~성적표에 메모리 감시 없음 · `run_full_v5.sh` 대기 한도 하드코딩 · 대시보드 "찍기보다 높음" 이 값 비교뿐~~ → **2026-09-26 고침**(아래 커밋): 성적표 에피소드마다 `Watchdog("scorecard")`(여유 < 25 % 면 끝난 에피소드 저장 · `aborted.json` · 끝 코드 3 · 프로세스 크기 문턱은 아직 안 재서 null) · 대기 간격 · 한도 `config/guard.yaml script_wait_*` · 판정 표에 "찍기 대비 p" 열과 "통과 · 찍기와 구별 안 됨"(v5 물체 쪽 p 0.145). 시험 96 통과.
6. 자료 v2 는 `retino-v1` 로 수집됨 — retino-v2(앞뒤 축)는 재주입 또는 v3 에서. (D 6)

---

## 6. 덧붙임 — 2026-09-26 ~ 29 (냄새 좌우 M0~M4 · train-v6)

> 이 W 를 쓴 뒤(09-26) 잔여 이슈 1 · 2(냄새 좌우 · 평가 자료)를 풀려고 한 일. 수치 정본은 R-7469 F-003 · F-005 인수 기준과 학습 안내서 10장 1.1 · 1.2c · 1.3 · 2.6 · 2.7.
> **다시 하는 입구는 `scripts/README.md`**(단계별 · 종류별 · 분석 6 · 09-29 실제 자료로 재현 확인).

### 6.1 변경 파일(커밋)

| 무엇 | 커밋 |
|---|---|
| 센서를 더듬이로(task-v2.1) · no_circuit 글도 설정을 따름 | `6da977f` |
| `odor-windows`(M2) · `--channel` · `--max-decision` | `9ac0214` · `efa0104` · `62837b2` |
| `inspect odor-control`(M1) · `odor-contrast`(M1b) | `e38348d` · `8c2fff5` |
| 재생 버그(직전 교사 행동) · 새 채널 검증 건너뜀 · `--seeds` · episodes · teacher_rules | `efa0104` · `338a252` · `9b969ec`(copy_episodes) |
| channels-v3(smell_ln) | `efa0104` |
| task-v4(냄새 맡기 · 4~7 mm · 옆 방위) · 실행 행동 기록 · STOP | `62837b2` · `abcb4ba` |
| stats-v3(자료 판으로 자동 · 냄새 쪽 칸 · 찍기 대비 p) | `e61d5de` · `cb3ccb2` |
| train-v6 · run_full_v6.sh(`--dataset` · 판정 · `--train-only`) | `cb3ccb2` · `66e5a8d` · `7f27e8a` |
| 성적표 결정 기록 scorecard-v3(R-5777 F-002) | `e069579` |
| 묶음 카드 재배선 씨앗 목록 | `56d4b73` |
| 재현 스크립트 | `280448c` · `0fda52b` |

### 6.2 시험 · 검사(원문)

```
$ scripts/00-check/check.sh        # 2026-09-29 07:06
ruff check src tests scripts → 끝 코드 0
pytest -q -m "not heavy"      → 끝 코드 0 (110 passed)
```

재생 검증(원본과 같은지 · 고친 뒤): `replay --pack fly-loop-v2 --split eval --rewired-seeds 0-4 --no-retune --verify 6` → 결정 0~5 모두 **최대 차 0**.

### 6.3 동작 확인 기록(파일 · 결과)

| 단계 | 파일 | 결과 |
|---|---|---|
| M2 | `runs/odor-windows-20260926-173621.json` | 옛 과제 · 투사 뉴런 — 누적해도 찍기 수준 |
| M1 | `runs/odor-control-20260926-175017.json` | PN 부호 1.000(차 2~4 %) · LN4 d′ 10~36(사후) · DNa02 0 Hz |
| M1b | `runs/odor-contrast-20260926-184048.json` | PN 20 %·3 s 에서야 · LN4 5~10 %·1 s 이상 |
| M3 | `runs/odor-windows-20260926-2037{26,33}.json`(fly-loop-v2-ln) | LN4 도 옛 과제에선 찍기와 구별 안 됨 |
| M4 작은 시험 | `runs/odor-windows-20260926-2231*.json`(fly-loop-v4-pilot) | 전체 LN4 10 누적 0.682 · 찍기 0.485 |
| M4 전체 | `runs/odor-windows-20260927-021030.json` 등 · `runs/linear-probe-20260927-021024.json`(fly-loop-v4) | **냄새 맡기 LN4 4 누적 0.725 · 재배선 0.525 · 찍기 0.490 · stats-v3 Holm 0.001** |
| v6 관문 · probe | `runs/compress-fly-loop-v4-compress-v3/gate.json` · `runs/train-probe-20260927-021708.jsonl` | 통과 시각 · smell_ln · 공간 · 31.3 GB |
| v6 판정 | `runs/caption-judge-train-v6-final.json`(`c999c6766e9cda7e`) · `-stage1-caption.json`(`c647768f86b70752`) | 물체 있음 · **냄새 쪽 통과**(0.530 · 캡션 단계본 0.577) |
| v6 성적표 | `runs/scorecard-20260928-124155/scorecard.json`(`c889e93a44fb2a17`) | vector 0.350 · 재배선 0.200/**0.375**/0.300 → **R 2.0-6 세 번째 발동** |

### 6.4 인수 기준 대조(덧붙임)

| 기능 | 성공 기준 | 결과 |
|---|---|---|
| F-003 | 후각 계열 채널 → 냄새 쪽이 찍기를 넘고 재배선 앙상블 상위 | ✅ **task-v4 · LN4 로 충족**(선형 검사 · 캡션 판정 둘 다) · 옛 과제 · 투사 뉴런으로는 불가 |
| F-005(v6) | 캡션이 찍기 · 재배선 앞섬 **그리고** 성적표가 재배선보다 높음 | ⚠️ 캡션만(물체 있음 · 냄새 쪽) · 성적표는 재배선 s1 에 짐 |

### 6.5 잔여 이슈(갱신)

1. ~~냄새 좌우~~ → 캡션에서 풀림 · **행동으로는 아직**(도는 쪽 같은 비율 진짜 0.558 대 재배선 0.508~0.580).
2. 행동 단계가 캡션을 깎는다 — v6 에서도(냄새 쪽 0.577 → 0.530) → 인수인계 D 4.
3. 성적표 씨앗 40 은 95 % 구간 ±0.15 — 0.35 대 0.375 는 가를 수 없다(판정력).
4. 조향 하강 뉴런 DNa02 침묵(M1) — 회로 쪽 한계 → F-006 생리 목표.
5. GPU 경합 — 다른 세션의 GPU 작업이 있으면 학습 2~3 배 느림 · GPU 오류(09-27 s1 재시작).

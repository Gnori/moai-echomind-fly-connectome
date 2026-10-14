# fly-learning-next — core 검증 보고 (W)

## 메타데이터

- **요구사항**: 7469 — [./20260923-104151-7469_fly-learning-next_req.md](./20260923-104151-7469_fly-learning-next_req.md)
- **대응 P**: [./20260923-104151-7469_fly-learning-next-core_dev-plan.md](./20260923-104151-7469_fly-learning-next-core_dev-plan.md)
- **관련 I**: [./20260923-104151-7469_fly-learning-next_interface-design.md](./20260923-104151-7469_fly-learning-next_interface-design.md)(v2 §7 F-006 · §8 F-007 · §9 retino-v2)
- **작성일**: 2026-10-14
- **범위**: core P 의 Task 1~7 = F-001(재배선 앙상블) · F-003(냄새 좌우 그래프 쪽) · F-008 2~3단계(시야 대응표 · 봉우리 검사) · **F-006(세포 종류 단위 회로 학습)** · **F-007(버섯체 가소성)** · retino-v2
- **결과**: **조건부 통과** — 코드 · 실행 · 결과 전 잠근 판정은 P 대로 끝났다.
  **진짜 배선 위에서 선언한 도파민 짝짓기 규칙이 냄새 특이적 · 구획 특이적 MBON 약화를 낸다**(결과 전 잠근 대조 여섯 통과)
  · 그 갈래만 섞은 배선(PN→KC · KC→MBON)보다 냄새 특이성이 높다(같은 매개변수) · 가치 읽기는 몸 없이 짝지은 냄새 쪽으로 옮긴다.
  그러나 ① 회로 학습이 맞춘 KC 희소성은 **재배선도 똑같이 맞춘다**(F-006 기준의 "실패로 기록" 쪽) ② **같은 절차 대조**(섞은 뇌마다 다시 맞춤) · **몸 행동(A2)** · 2단계 대리 기울기 · 공간 목표는 하지 않았다(성공 가능성 낮음 · 요구자 10-14 *"성공 가능성이 낮으면 하지마"*).
  → "초파리 뇌가 배운다" · "버섯체 출력이 행동을 정했다" · "섞은 배선(뇌 전체)보다 낫다"는 **쓰지 않는다**.

> 원칙: 서술은 증거가 아니다 — 아래 로그 · 파일 · sha256 만 증거다. 파일은 `runs/`(git 밖 · 백업 `~/Backups/moai-echomind-fly-connectome`).
> 숫자의 정본은 P(core 계획서)다 — 여기는 판정에 쓰인 것만 옮긴다.

---

## 1. 변경 파일 목록

| 파일 | 변경 유형 | 요약 |
|---|---|---|
| `src/moai_echomind_fly/graph.py` | 수정 | `--rewired-seeds` 범위 · **부분 섞기 `rewire_subset` · `brain_graph`(`pnkc-s*` · `kcmbon-s*`)** — 한 갈래만 도착을 섞고 나머지는 진짜 그대로 |
| `src/moai_echomind_fly/sim.py` · `config/sim.yaml` | 수정 | sim-v2(`ipsi_release` · F-003) · `Engine(cell=)` · `set_edge_weights` |
| `src/moai_echomind_fly/inspect.py` | 추가 · 수정 | `inspect laterality` · `inspect bump` |
| `src/moai_echomind_fly/retino.py` · `config/retino.yaml` | 추가 | retino-v1(고도) → retino-v2(방위 + 고도 · 뒤섞은 대응표) |
| `src/moai_echomind_fly/cellparams.py` | 추가 | sim-v3 매개변수 파일 `CellParams`(종류별 τ · 문턱 · 묶음 배율 α · 정체성 검사) |
| `src/moai_echomind_fly/circuit_fit.py` · `config/fit.yaml` · `config/physiology.yaml` | 추가 | CMA-ES 1단계 · 생리 손실 physiology-v1 → **v7** · 범위 escape-small · kc-sparse-a/b/c · kc-mbon-a/b/c/al · `init_from` · `fixed_alpha` · `targets_scope` · 촉각엽 `al_fix` |
| `src/moai_echomind_fly/circuit_grad.py` | 추가 | 2단계 엔진 `FuncLIF`(대리 기울기 · 체크포인트) — **학습 고리 · 손실은 만들지 않음** |
| `src/moai_echomind_fly/plasticity.py` · `config/plasticity.yaml` | 추가 | plasticity-v1(구획 표 · 3요소 규칙 · 대조 · `--detail`) · η 보정 calib-a · 부분 섞기 판정 pr-a · 가치 읽기 vr-a |
| `src/moai_echomind_fly/mb_probe.py` · `mbon_probe.py` | 추가 | KC 희소성 측정 M0 · MBON 침묵 진단 |
| `src/moai_echomind_fly/cli.py` | 수정 | `fit-probe` · `fit` · `fit-eval` · `kc-probe` · `mbon-probe` · `plasticity-probe` · `plasticity-controls` · `plasticity-cs-screen` · `plasticity-eta-calibrate` · `retinotopy` |
| `scripts/20-odor/*` · `scripts/30-train/fit*.sh` · `scripts/50-analysis/*` | 추가 | 실행 · 재기 · 분석 스크립트(사건 파일 → 대시보드) |
| `tests/test_{cellparams,circuit_fit,circuit_grad,plasticity,mb_probe,mbon_probe,retino,retino_v2,sim_ipsi}.py` | 추가 | 아래 2장 |

커밋(core 주요): `8964835`(F-003 sim-v2) · `14eef3b` · `58a1569`(retino-v1 · v2) · `3a3a323`(시험 넷) ·
`8e7c270`(physiology-v1) · `09df530`(CellParams) · `f3f2e0a`(CMA-ES · fit) · `9f4abce`(FuncLIF) · `2d47620`(plasticity-v1) · `bc0b0b4` · `06b146a` · `d619f31` ·
`fe28734` · `701c1ef` · `5a374ae` · `d9eeda3` · `9661a54` · `b89903c` · `7bf33f2` · `a7302b6` · `84bdc83` · `05a5c81` · `f4a5b34` · `aa0c6d6` · `f8869a3`(결과 전 잠금은 이 가운데 `fe28734` · `701c1ef` · `5a374ae` · `d9eeda3` · `9661a54` · `b89903c` · `7bf33f2` · `a7302b6` · `84bdc83` · `f4a5b34` · `aa0c6d6`).

---

## 2. 테스트 실행 로그 (원문 · 2026-10-14 21:4x)

```
$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q -m "not heavy"
197 passed, 3 deselected, 18 warnings in 13.45s

$ uv run pytest -q -m "not heavy" tests/test_cellparams.py tests/test_circuit_grad.py tests/test_circuit_fit.py \
    tests/test_plasticity.py tests/test_mb_probe.py tests/test_mbon_probe.py tests/test_retino_v2.py \
    tests/test_sim_ipsi.py tests/test_loop_parts.py
55 passed in 2.41s
```

판정에 기대는 시험: `test_cellparams`(처음 값이면 발화 완전 일치 · 범위 밖 불변) · `test_circuit_grad`(대리 기울기를 켜도 정방향 발화 완전 일치) ·
`test_plasticity::test_pairing_weakens_only_paired_kc_in_the_dan_compartment` · `::test_no_change_without_dan_or_with_zero_eta` ·
`::test_apply_scales_engine_weights_on_kc_mbon_edges_only` · `::test_rewire_subset_keeps_other_edges_and_degrees` · `test_retino::test_weighted_input_hex_is_synapse_weighted_mean`.

---

## 3. 모듈 동작 확인 기록

### 3.1 F-001 · F-003 · F-008 2~3 (Task 1~4 · 7)

| 무엇 | 결과 | 근거 |
|---|---|---|
| 재배선 앙상블 | `graph --rewired-seeds 0-19` · 메타에 씨앗 · 규칙 · 자기 연결 수 | P Task 1 |
| 냄새 좌우(그래프) | ORN → PN 같은 쪽 51.7 % · 반대쪽 47.7 % → sim-v2(같은 쪽 × 1.4 · Gaudry 2013) · 실제 입력 부호 0.508 그대로 → **냄새 쪽 칸 판정 제외** | `runs/inspect-laterality-20260923-105740.json` |
| 시야 대응표 | retino-v1(고도) → retino-v2(방위 · 앞뒤 축은 LC4 → DNp02/DNp11 기울기로 독립 확인) · 뒤섞은 대응표 | `58a1569` · 연구 노트 13 §4 |
| 시야 대응표 대조(진짜 · 눈 안 섞음 · 두 눈 섞음) | 물체 쪽 0.794 · 0.797 · 0.775 · 찍기 0.667 · p 0.064 → **못 넘음**(물체 쪽은 좌우 눈으로 실림) | P Task 7 · `data/datasets/fly-loop-v5-ret-*` |
| 중심복합체 봉우리 | 진짜 EPG 발화 0 / 3,297 결정 → **봉우리 없음**(예상된 한계 · 쉬는 상태 침묵) | `runs/bump-check-20260923-195413.json` |

### 3.2 F-006 회로 학습 (Task 5)

| 판 · 범위 | 진짜 | 재배선 s0(같은 절차) | 읽기 |
|---|---|---|---|
| escape-small | 처음 손실 이미 0(도망 5 항목 만족) → **돌리지 않음** | — | 배울 것이 없는 목표 |
| kc-sparse-b(24 · 13 × 50) | 1.77 → 0.079 · 식초 KC 1.0 → **4.8 %** | 2.09 → 0.213 · 0.44 → **4.7 %** | **희소성은 배선 특이적이 아님** · 진짜 일반화 2/5 대 재배선 4/5 |
| kc-sparse-c(엽 조건) | 5.79 → 0.073 · MBON 3/97 | 5.03 → 0.233 · 0/97 | 엽 조건도 둘 다 맞춤 · MBON 여전히 침묵 |
| kc-mbon-a(KC→MBON 묶음) | 13.5 → 0.041 · **MBON 32/97** | → 10.09 · **0/97** | MBON 냄새 반응은 **진짜에서만** — 까닭은 학습 전부터 있던 KC→MBON 수렴(MBON 당 KC 시냅스 중앙 2,502 대 71) |
| kc-mbon-b(냄새 넷) | 24.7 → 0.834 · 일반화 5/5 | → 13.8 · 0/97 | MBON 반응 좋아짐 · 냄새 특이성 없음 |
| kc-mbon-c(KC 겹침 상한) | ❌ 20/50 세대 중단 | 안 돌림 | 벌점이 KC 를 끈 해를 이기게 한 설계 결함(`958eef7`) |
| kc-mbon-al(촉각엽 흥분성 LN → PN ×0.5 고정) | 12.4 → **0.535** · KC 2.6~6.1 % · MBON 59 % · 발톱 0.98 · 도망 그대로 | 13.4 → 13.3 · MBON 0.7 % | params `703a805180e0e950` · 재배선 `bf6c382ab2a4f714` |
| 같은 절차 판 작은 시험(pnkc-s0 · 5 세대) | (진짜 같은 5 세대 1.67) | 11.4 → **4.92**(기준 ≤ 3 못 넘음) | 본 학습 안 함(`f3cbe4f`) |

공통: 연결 가중치 불변(그래프 해시 전후 같음) · 도망 반사 재채점 그대로(GF 0.60 · 36 ms) · 학습 매개변수는 종류별 τ · 문턱 · 묶음 배율만(kc-mbon-al 25).
측정(무거운 실행 규약): `fit-probe` 시행 2 · 4 → 시행당 약 3.3 s + 고정 5.8 s · 최대 2.3 GB(`runs/fit-probe-escape-small-real-20261003-071348.json`) · kc-sparse-b 평가 한 번 16.9 s · 2.2 GB.

### 3.3 F-007 버섯체 가소성 (Task 6)

규칙 plasticity-v1(KC 흔적 × 구획 DAN 신호 · 곱셈 약화 · m ∈ [m_min, 1]) · 구획 37 · 문헌 대조 7/7 · KC → MBON 61,210 연결 · η 는 Hige 2015 감소폭(50~90 %)으로 보정 → **0.002**(calib-a · 판정과 다른 씨앗 · CS+ 만).

**대조 여섯(kc-mbon-al 진짜 · 씨앗 0~4 · 결과 전 기준 `5a374ae` · `84bdc83`)** — `runs/plasticity-controls-real-20261013-205535.json`(sha256 `06951a09654d42a1…`)

| 조건 | 보상 구획 CS+ | 처벌 구획 CS+ | 보상 구획 CS− | 판정 |
|---|---|---|---|---|
| 짝짓기 | **−71 %**(5/5) | +2 % | **−19 %** | ✅ · ✅ |
| 어긋난 보상 · 표 섞기 · 끔 | −1 · −19 · −1 % | −9 · −19 · 0 % | −1 · −7 · 0 % | ✅(표 섞기는 경계에 가까움) |
| 처벌 DAN | −7 % | **−85 %** | −3 % | ✅(뒤집힘) |

**섞은 배선 대조 — 부분 섞기 pr-a(씨앗 0~9 · 같은 매개변수 · 결과 전 `f4a5b34`)** — `runs/partial-rewire-judge-20261013-213629.json`(sha256 `7a3b4f25676d444b…`)

| 뇌 | 특이성 SI | 짝짓기 때 처벌 구획 | 판정 |
|---|---|---|---|
| 진짜 | **0.513** | +2.5 % | — |
| PN→KC 섞기 s0 · s1 · s2 | 0.351 · 0.410 · 0.487 | −23~−28 % | ✅ p 0.0020 |
| KC→MBON 섞기 s0 · s1 · s2 | 0.463 · 0.480 · 0.471 | −1~+2 % | ✅ p 0.0098(차 작음) |
| 전체 섞기(재배선 s0) | MBON 침묵 | — | 잴 수 없음 |

**가치 읽기 A1(몸 없이 · 결과 전 `aa0c6d6` vr-a)** — `runs/valence-judge-20261014-075726.json`(sha256 `76c6212923c149ce…`): 보상 짝짓기 ΔV **+114.1**(10/10 · p 0.002) · 처벌 **−124.0** · 어긋난 · 표 섞기 · 끔 −12.4 · −20.9 · −1.3 · 무작위 부호 20 벌 모두 아래 → 주 기준 넷 통과.
보조: PN→KC 섞기(67~84)보다 큼 ✅ · KC→MBON 섞기(122~162)보다 작음 ❌.
결과 뒤 탐색(판정 아님 · `runs/valence-norm-20261014-212340.json` sha256 `b7a4a1edc95f41fb…`): KC→MBON 섞은 뇌는 MBON 반응이 1.4 배 — 크기로 나누면 구별 안 됨(p 0.057) · 진짜 대 섞은 배선 순서는 **읽는 법마다 뒤집힌다**.

### 3.4 인수 기준 대조 (R-7469 §2.2 · core 쪽)

| 기능 | 성공 기준 · 인수 기준 | 결과 |
|---|---|---|
| F-001(core) | 재배선 앙상블을 쉽게 | ✅ |
| F-003(core) | 같은 쪽 / 반대쪽 측정 · 판 올림 | ✅(고칠 수 없음을 기록 · 냄새 쪽 판정 제외) |
| F-008 2~3 | 시야 대응표 · 봉우리 검사 | ✅ 구현 · 봉우리는 **실패로 기록** · 대응표 대조는 못 넘음 · 4~5 단계 안 함 |
| F-006 | 매개변수 목록 · 판 · 가중치 불변 / 작은 측정 / 진짜 · 재배선 표 · 도망 재채점 | ✅ 셋 다 기록 |
| F-006 | **같은 학습을 재배선에도 · 학습 뒤 진짜 > 재배선 앙상블** | ⚠️ 재배선은 **s0 하나** · KC 희소성은 둘 다 맞춤(→ 기준의 "실패로 기록") · MBON 반응은 진짜만(배선 성질) · 2단계 · 공간 목표 안 함 |
| F-007 | 규칙 · 출처 · 학습률 기록 / 세 대조군 비교표 | ✅(plasticity-v1 · η 0.002 · 대조 여섯 · 부분 섞기 둘) |
| F-007 | 학습 뒤 CS+/CS− 해독이 찍기를 넘고 ①②③ 모두보다 높고 ④에서 뒤집힘 | ⚠️ **구현 중 바뀜** — `value` 채널 선형 해독(task-v3) 대신 MBON 반응 감소로 잼 · ① 끔 ✅ ② 무작위 보상 → 어긋난 보상 ✅ ③ PN→KC 섞기 → 부분 섞기(같은 매개변수) ✅ ④ 처벌 DAN 뒤집힘 · 표 섞기 ✅ · 행동(2차)은 A1 까지 · A2 하지 않음 |

---

## 4. 정본 문서 동기화 내역

| 문서 | 내용 | 커밋 |
|---|---|---|
| 연구 노트 11 §8.13 · 안내서 03 §6.4 · §6.5 · 05 · 00 · 02 | 촉각엽 고침 · η 보정 · 대조 여섯 · 부분 섞기 pr-a(doc 세션) | `094b8f9` · `d1f5ec7` |
| 연구 노트 13 §7 · 16 §8~§11 | MBON → 행동 읽기 · KC 희소성 · KC 겹침 · 촉각엽 번짐(research 세션) | `ba6b4cb` · `2e1e471` · `e320d73` · `bf9b3aa` |
| 학습 안내서 10장 대장 | kc-sparse · kc-mbon 회로 학습 손실 · params sha256 | (회로 학습마다) |
| `docs/model-v1.md` | 1판 정리 · 쓸 수 있는 말 | `3b24e56` |
| CLAUDE.md · 인수인계 | 10-14 상태 · 하지 않은 것과 까닭 | `e5edee0` · `db1e309` |
| ⏳ 진행 중(doc 세션 일감 10-14) | A1 · 같은 절차 작은 시험 → 노트 11 · 안내서 03 · 05 · 부분 섞기 → 안내서 00 · 02 · README · 명령 모음 11장 | 회신 뒤 이 줄에 해시 |

---

## 5. 잔여 이슈

1. **같은 절차 대조 없음** — 부분 섞기는 진짜로 맞춘 매개변수를 그대로 얹었다. 섞은 뇌마다 다시 맞추는 판은 작은 시험에서 기준을 못 넘어(4.92 대 ≤ 3) 하지 않음 — 성공 가능성 낮음(요구자 10-14).
2. **몸 행동(A2) 없음** — 방향 · 이득을 손으로 정하는 구조라 몸 결과 순서가 A1 정규화 ΔV 순서를 따르고, 그 순서가 읽는 법마다 뒤집혀 "섞은 배선을 둘 다 넘음" 가능성이 낮음 → 하지 않음. 다시 열 조건: 회로가 방향을 주는 길(DNa02 가 냄새에 반응).
3. **모형 수정이 판정 앞에 있다** — 촉각엽 흥분성 LN → PN ×0.5(근거 노트 16 §11 · 실제는 주로 전기 시냅스) · η · CS− · 회로 학습 모두 진짜 배선에서 정함.
4. **구획 특이성의 상당 부분은 규칙 설계에서 온다**(그래프의 DAN → MBON 표를 쓰게 만듦).
5. **F-006 2단계(대리 기울기) · F-008 4~5 단계 · retino-v2 로 학습** — 엔진만 있고 학습 고리 · 손실 없음 · 착수 안 함.
6. 전체 섞기 재배선은 MBON 이 침묵해 F-007 대조가 되지 못한다 — 부분 섞기로 대신했다.

# fly-learning-next — core 구현 플랜 (P)

## 헤더

- **요구사항**: 7469 — [./20260923-104151-7469_fly-learning-next_req.md](./20260923-104151-7469_fly-learning-next_req.md)
- **인터페이스 계약(I)**: [./20260923-104151-7469_fly-learning-next_interface-design.md](./20260923-104151-7469_fly-learning-next_interface-design.md) (승인 v1 · 구현 중 변경 위임)
- **모듈**: core — `graph.py` · `sim.py` · 새 분석(`inspect` · `retinotopy` · `bump`) · `config/retino.yaml`
- **작성일**: 2026-09-23
- **Goal**: 재배선 앙상블을 쉽게 만들고, 감각 입구의 결함(냄새 좌우)과 3차원 재료(LC 시야 자리 · 중심복합체 봉우리)를 **그래프와 뇌만으로** 잰다. 회로 학습(F-006 · F-007)은 I v2 뒤.
- **선행 P 의존성**: 없음 — loop P 의 Task 2 · 3 이 이것에 기댄다

---

## Task 1: 재배선 앙상블 (F-001)

- [x] `graph --rewired-seeds 0-19`(범위) · 만든 그래프 메타에 씨앗 · 규칙 · 자기 연결 수

## Task 2: 냄새 좌우 측정 (F-003)

- [x] `inspect laterality` — 단계별 같은 쪽 / 반대쪽 비율(CPU 0.8 s): 1단계 51.7 / 47.7 · 2단계 72.6 / 25.6 · 3단계 76.8 / 21.9
- [x] 고칠 판 = **`sim-v2`**(`ipsi_release` · Gaudry 2013 · 옛 판 `config/archive/sim-v1.yaml`) · 시험 `tests/test_sim_ipsi.py`
      · 효과 측정: 극단 입력(150 대 60 Hz)은 같은 쪽이 높아짐 · 실제 입력(65 쌍)은 부호 0.508 그대로 → 냄새 쪽 칸 판정 제외(stats-v2)

## Task 3: LC 시야 대응 retino-v1 (F-008 2단계)

- [x] `config/retino.yaml`(규칙 · 출처 · 방향 근거) · `retinotopy build`(CPU 1 s) — 시각 투사 4,096 개 고도 · missing 5,105
      · 위 · 아래 축 확인: R7d/R8d → 기둥 y ≈ 64 · HSN 47 > HSE 40 > HSS 25 · **앞 · 뒤 축은 미확정**(방위는 눈 단위)
      · LC4 126/126 · LPLC2 185/185 · LC16 182/182 · LPLC1 134/134 · **LC17 9/353 · LC6 5/124**(다른 소엽 뉴런에게서 받음)
- [ ] 시험: 작은 가짜 그래프에서 가중 평균이 맞다 · 뒤섞은 대응표 생성

## Task 4: 중심복합체 봉우리 검사 (F-008 3단계)

- [x] `inspect bump`(구현 중 변경 — 뇌만 돌리는 대신 **자료 v2 로**: 공간 채널 EPG 구획 위상 대 참값 몸 방향 · 원형 상관 · 에피소드 단위 순열 · 진짜 대 재배선) · 시험
      · "단서를 끈 뒤 유지"는 자료로 못 잰다 — 필요하면 뇌만 검사를 따로
- [ ] 결과(통과든 실패든) 기록 → F-006 목표에 반영

## Task 5: F-006 회로 학습 (I v2 §7 · 착수는 F-005 판정 뒤 · 무거운 실행은 요구자 확인)

- [x] `config/physiology.yaml`(physiology-v1 · 11 항목 · 출처 · 허용폭 · 확인 수준 — `8e7c270`)
- [x] sim-v3 매개변수 파일 형식(2026-10-01 · 미리 개발) — `cellparams.py`(`CellParams` · `init` · `init_scope` · 정체성 검사 `check` · 저장/불러오기) · `config/fit.yaml`(범위 이름 · 경계) · `Engine(cell=)` 선택 인자(없으면 지금 길 그대로). 시험 7: **처음 값이면 발화 완전 일치**(작은 그래프) · α 가 범위 안 연결만 · tau · 문턱은 범위 안 뉴런만 · 두 시간 상수가 같을 때 극한 연속 · 음수 α 거부.
  실측(traced-v1): escape-small 17 종 · 뉴런 2,534 · 범위 안 쌍 전부(≥ 100 시냅스) 85 → 매개변수 119(목표 ≤ 50 넘음) → **제안: 출력 쌍(→ DNp01/02/04/11)만 10 · 매개변수 44 · 연결 1,060** — ⚠️ 무엇을 배우나를 정하는 선택이라 착수 전 요구자 확인. sim.yaml 은 그대로(sim-v2 · 도는 성적표가 읽음)
- [ ] 함수형 걸음 + 대리 기울기 + `grad_scope` · 시험: 대리 기울기 끄면 지금 엔진과 발화 완전 일치(작은 그래프)
- [ ] `fit-probe --scope escape-small --window-ms 10,30` → 비례 확인 → **요구자 확인** → `fit --method cmaes`(1단계) → 재배선 s0~s4 같은 절차 → 정체성 검사 · 도망 재채점
- [ ] 2단계 `sgbptt`(space · escape) — 같은 순서

## Task 6: F-007 버섯체 가소성 (I v2 §8 · Task 5 뒤)

- [x] `config/plasticity.yaml`(plasticity-v1 · 규칙 · η 보정 목표 · 구획 · 사건 → DAN · 대조군 다섯)
- [ ] 구획 표를 그래프에서 생성 · `plasticity-probe`(KC 발화 비율 · MBON 반응 · DAN 효과 · η 보정 — 몸 없이 창 몇 개)
- [ ] `task-v3`(냄새 둘) · 대조군 다섯 · `value` 채널 해독(stats-v2)

## Task 7: retino-v2 (I v2 §9 · 가벼움)

- [x] `config/retino.yaml` → retino-v2(h · v · 방위 [−170, +10] 선언 · 왼쪽 거울) · `retinotopy build` · 뒤섞은 대응표 둘 — research 세션 `58a1569` · 부호 검증 LC4→DNp02 +0.76/+0.85
- [x] 재주입 `replay`(몸 없이 뇌만 · 조각별 씨앗 순서 · `--verify`) · `retinotopic2` 인코더 — `a94b182`(⚠️ 수집이 끝날 때까지 config/retino.yaml 은 v1 로 되돌려 둠 · task.yaml 도 그대로)
- [ ] 자료 v2 재주입(`collect --replay`) 또는 v3 — 작게 잰 뒤 요구자 확인
- [x] 대조군 준비(2026-10-01) — `build_map(retino_rule=)` · `Loop(retino_rule=)` · `replay --retino-rule` : 뒤섞은 대응표로 시야를 조율해 **같은 절차의 대조군 자료**를 만든다(자료 메타 `versions.retino` · `replay_of.retino_rule` 에 판이 남음). 조율 모드가 아니면 거부 · 시험 1(자극 뉴런 순서 같음 · 방위만 눈 안에서 섞임)
- [ ] 순서(성적표가 끝난 뒤 · 설정 파일은 그 전에 안 만짐): ① task-v5 = `vision.mode: retinotopic2` + `azimuth_sigma_deg`(선언 · 판 올림 · 옛 판 archive) ② `replay --retune` 으로 fly-loop-v4 → 새 묶음(진짜) ③ 같은 원본을 `--retino-rule retino-v2.shuffled-azimuth-s0`(눈 안) · `…-type-s0`(두 눈) 으로 재주입 ④ 물체 방향 선형 검사: 진짜 > 섞은 대응표일 때만 "시야 자리가 쓰였다" ⑤ 학습은 그 뒤 판단 — 무거운 단계마다 요구자 확인

## 최종 검증

- ruff · 가벼운 시험 · 각 Task 의 기록을 W 에

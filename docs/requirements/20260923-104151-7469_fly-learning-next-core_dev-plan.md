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

- [ ] `config/retino.yaml`(규칙 · 출처) · `retinotopy build` — LC/LPLC 마다 입력 기둥 뉴런(`assignedOlHex1/2`)의 시냅스 가중 평균 → 시야 자리 · 못 얻은 뉴런은 `missing`
- [ ] 시험: 작은 가짜 그래프에서 가중 평균이 맞다 · 뒤섞은 대응표 생성

## Task 4: 중심복합체 봉우리 검사 (F-008 3단계)

- [ ] `bump-check` — 뇌만 · 시야 자리별 입력을 돌려 방향 회전 흉내 · EPG 구획 위상 · 원형 상관 · 단서 끈 뒤 유지 · 진짜 대 재배선
- [ ] 결과(통과든 실패든) 기록 → F-006 목표에 반영

## 최종 검증

- ruff · 가벼운 시험 · 각 Task 의 기록을 W 에

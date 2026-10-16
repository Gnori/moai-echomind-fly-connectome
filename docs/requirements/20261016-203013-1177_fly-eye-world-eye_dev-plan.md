# fly-eye-world — eye 구현 플랜 (P)

## 헤더

- **요구사항**: 1177 — [./20261016-203013-1177_fly-eye-world_req.md](./20261016-203013-1177_fly-eye-world_req.md)
- **인터페이스 계약(I)**: [./20261016-203013-1177_fly-eye-world_interface-design.md](./20261016-203013-1177_fly-eye-world_interface-design.md)(v0.2 · 승인 10-16)
- **모듈**: eye
- **작성일**: 2026-10-16
- **Goal**: 그림(큐브맵) → 낱눈 · 입구 뉴런마다 밝기 → 앞단 v2(T4/T5 국소 움직임 · Tm2 · Tm4 · T2 · Tm3) 발화율. 분석식(V0 · V0b)과 같은 꼴의 출력.
- **Architecture**: `eye/compound.py`(방향 → 큐브맵 무게 희소 행렬 · 기둥 격자 이웃) · `eye/frontend.py`(ON/OFF · 이웃 지연 곱 4 방향 · 소엽 종류 dL/dt) · 기존 `geometry.py` · `disk.py`
- **Tech Stack**: numpy · scipy 없이(희소는 인덱스 · 무게 배열로)
- **선행 P 의존성**: 없음(렌더러 없이 합성 큐브맵으로 시험)

> 이 저장소의 계획서 형식: Task 마다 파일 · 할 일 · 검증(Run · Expected). 코드는 소스가 정본. 진행 정본은 R 상태.

---

## Task 1: 겹눈 표본(compound)
**Files**: `src/moai_echomind_fly/eye/compound.py` · `tests/test_eye_compound.py`
- [ ] 방향(방위 · 고도) → 큐브맵 면 · 픽셀 · 가우시안 받아들이는 각(σ) 무게(미리 계산) · 밝기 = Σ 무게 × 픽셀
- [ ] 기둥 격자: L1 공식 좌표(눈마다 875 · 892) → 이웃(가로 Δh ±2 · 세로 대각 둘)
- Run: `uv run pytest -q tests/test_eye_compound.py` · Expected: 한 방향만 밝은 큐브맵 → 그 방향 낱눈이 최대 · 같은 그림 → 비트 같음

## Task 2: 앞단 v2(frontend)
**Files**: `src/moai_echomind_fly/eye/frontend.py` · `config/eyeworld.yaml frontend` · `tests/test_eye_frontend.py`
- [ ] 기둥 밝기 시계열 → 시간 고역 → ON/OFF → 이웃 지연 곱(τ · 문헌 · 결과 전 잠금) 4 방향 → T4(ON) · T5(OFF) 발화율(기둥 → 뉴런 = 입력 가중 자리의 가장 가까운 기둥)
- [ ] Tm2 · Tm4 · T2 · Tm3 — V0b 와 같은 식(뉴런 자리에서 흐린 밝기의 dL/dt · 저역 · 기준)
- Run: `uv run pytest -q tests/test_eye_frontend.py` · Expected: 줄무늬를 한 방향으로 움직이면 그 방향 T4/T5 만 · 어두운 원판이 커지면 그 자리 Tm2 · Tm4

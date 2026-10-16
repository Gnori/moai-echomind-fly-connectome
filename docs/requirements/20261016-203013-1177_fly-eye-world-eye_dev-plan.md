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

## Task 3: 앞단 v3 · v3b(대비 적응 · 10-17 추가)
**Files**: `src/moai_echomind_fly/eye/frontend.py`(`Weber` · `DivNorm` · `line_v3` · `FrontendV3` · v2 는 훅만 나눔) · `config/eyeworld.yaml frontend_v3` · `tests/test_eye_frontend_v3.py` · `scripts/60-eyeworld/f0_frontend.py`
- [x] 까닭: V1-b T4/T5 평균 0.14 Hz — 상관 출력이 대비 제곱(대비 0.2 에서 기준의 0.04 · 실제 T4 약 0.81 · research 노트 18)
- [x] (가) 국소 웨버 대비(FWHM 25°) · (나) 정류 뒤 나눗셈 정규화(풀 FWHM 30° · p 1.2 · w 2 · c50) — 노트 18 §6.2 그대로 · 결과 전 잠금 `2c0aa85`
- [x] F0(앞단만 · 뇌 없음) 잠금 f0-a → d: F0-1 대비 곡선 · F0-2 가로 대비(보고) · F0-3 렌더 제자리 돎 · F0-4 좌우
- [x] F0-1 ✗ → 한 번 다시 고르기(fit-a · c50_contrast 0.05 · v3b) → ✓(보정 · 독립 증거 아님)
- [x] F0-4 가 렌더러 · 세계 버그 둘을 찾음(`f0e3a2c` · `5e49709`) → ✓ 1.00
- [ ] F0-3 ✗(arena-b 12.9 Hz · 기준 20 추측) → V0d 뇌 쪽 필요 입력량(`v0_dose.py` · 잠금 `6498acc`)으로 기준 다시 정함 → 요구자께 여쭘
- Run: `uv run pytest -q tests/test_eye_frontend_v3.py tests/test_eye_frontend.py` · `bash scripts/60-eyeworld/f0-frontend.sh` · Expected: 시험 4 통과 · F0 결과는 R-1177 F-001 표

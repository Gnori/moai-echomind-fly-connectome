# fly-eye-world — world 구현 플랜 (P)

## 헤더

- **요구사항**: 1177 — [./20261016-203013-1177_fly-eye-world_req.md](./20261016-203013-1177_fly-eye-world_req.md)
- **인터페이스 계약(I)**: [./20261016-203013-1177_fly-eye-world_interface-design.md](./20261016-203013-1177_fly-eye-world_interface-design.md)(v0.2 · 승인 10-16)
- **모듈**: world
- **작성일**: 2026-10-16
- **Goal**: 파이썬 세계 서버 — 개체 · 몸 · 충돌 · 장면 · 시계 · 기록 · 안전. 렌더러 없이도 돌고 같은 씨앗이면 같은 궤적.
- **Architecture**: `eyeworld/world.py`(Entity · Body · World.step) · `scene.py`(장면 YAML) · `safety.py`(금지 포트 · 프로필 · PID · 헤드리스 잠금 · 읽기 전용) · 설계 문서 2.7 · 2.9
- **Tech Stack**: 파이썬 3.14 · numpy · pyyaml(기존)
- **선행 P 의존성**: 없음

> 이 저장소의 계획서 형식: Task 마다 파일 · 할 일 · 검증(Run · Expected). 코드는 소스가 정본. 진행 정본은 R 상태.

---

## Task 1: 안전(safety)
**Files**: `src/moai_echomind_fly/eyeworld/safety.py` · `config/eyeworld.yaml safety` · `tests/test_eyeworld_safety.py`
- [x] 금지 포트 확인(설정 목록 · 우리 포트가 그 안이면 `SafetyError`) · 포트가 비었나(바인드 시험)
- [x] 에코월드 헤드리스 잠금(`/tmp/echomind-world-headless.lock`)이 살아 있으면(PID 살아 있음) 시작 거부 · 읽기만
- [x] 우리 프로필 폴더(runs 아래) · 띄운 PID 기록 · **그 PID(와 자식)만** 종료 · 이름 패턴 종료 없음
- [x] 에코월드 저장소 쓰기 0 — 실행 전후 표본 파일 수정 시각 비교 도구
- Run: `uv run pytest -q tests/test_eyeworld_safety.py` · Expected: 금지 포트 거부 · 잠금 흉내 거부 · 가짜 프로세스 PID 만 종료

## Task 2: 세계(world · scene)
**Files**: `src/moai_echomind_fly/eyeworld/world.py` · `scene.py` · `config/eyeworld_scenes/arena-a.yaml` · `tests/test_eyeworld_world.py`
- [x] `Entity`(id · kind · asset · pos m · yaw · vel · collider 원기둥 r · h · solid · anim) · `Body`(반경 0.35 · 키 · 눈높이 = 키 × 0.936 · 걷기 1.3 m/s · 달리기 · 회전 한계 · 비켜서기)
- [x] `World.step(dt, commands)` — 운동학(속도 · 회전 상한) · 원기둥 충돌(밀어냄 · 닿음 목록) · 움직임 대본(걷는 사람 · 도는 경기장) · 시계(고정 dt)
- [x] 장면 YAML(에셋 키 · 자리 · 크기 · 씨앗) → 개체 · 에셋 허가 칸(원본 기록에서 읽어 기록 · 허가 모름은 거부 · Mixamo 는 승인 목록)
- [x] 기록: 걸음마다 자세 묶음(`poses()` — 렌더러 통로 계약 2.6)
- Run: `uv run pytest -q tests/test_eyeworld_world.py` · Expected: 같은 씨앗 · 같은 명령 → 같은 궤적(비트) · 충돌 시 겹치지 않음 · 대본대로 움직임

# fly-eye-world — loop 구현 플랜 (P)

## 헤더

- **요구사항**: 1177 — [./20261016-203013-1177_fly-eye-world_req.md](./20261016-203013-1177_fly-eye-world_req.md)
- **인터페이스 계약(I)**: [./20261016-203013-1177_fly-eye-world_interface-design.md](./20261016-203013-1177_fly-eye-world_interface-design.md)(v0.2 · 승인 10-16)
- **모듈**: loop
- **작성일**: 2026-10-16
- **Goal**: 한 걸음씩 닫힌 고리 — 세계 → 렌더 → 겹눈 → 앞단 → 뇌 → 디코더 → 몸. V1 판정 실험(그림에서도 V0 · V0b 와 같은 판정) · 비용 · 메모리 실측.
- **Architecture**: `loop/eyeloop.py`(순서 · 기록 · 감시 · 실패 시 저장하고 멈춤) · `scripts/60-eyeworld/v1-*.sh`(사건 파일)
- **Tech Stack**: 기존 `sim.Engine` · `loop/decoder` · guard · Watchdog
- **선행 P 의존성**: world · render · eye

> 이 저장소의 계획서 형식: Task 마다 파일 · 할 일 · 검증(Run · Expected). 코드는 소스가 정본. 진행 정본은 R 상태.

---

## Task 1: 실행기(eyeloop)
**Files**: `src/moai_echomind_fly/loop/eyeloop.py` · `tests/test_eyeloop.py`(가짜 렌더러)
- [ ] 걸음: 명령 → 세계 dt → 자세 → 렌더 → 감각 → 앞단 → `set_rates` → `run(dt)` → 결정 창이 차면 디코더 → 기록(`decisions.jsonl` · `status.json` · `meta.json` 판 · 해시)
- [ ] 무거운 실행 잠금 · Watchdog · 크롬 PID 정리(안전 모듈)
- Run: `uv run pytest -q tests/test_eyeloop.py` · Expected: 가짜 렌더러로 끝까지 · 기록 꼴

## Task 2: V1 판정 실험(결과 전 잠금)
**Files**: `config/eyeworld.yaml v1` · `scripts/60-eyeworld/v1-check.sh`
- [ ] (가) 경기장 회전(몸 둘레로 장면이 돎) → DNa02 부호 · (나) 어두운 물체(바위 · 사람)가 앞-왼/오에서 다가옴 → 같은 쪽 LC4 — 진짜 · 자리 섞기 · 재배선 · 씨앗 · 기준은 V0 · V0b 와 같은 꼴
- [ ] 비용 · 메모리 표(한 걸음 · 렌더 · 뇌)
- Run: `scripts/60-eyeworld/v1-check.sh` · Expected: 판정 JSON · 대장(안내서 10장) 한 줄

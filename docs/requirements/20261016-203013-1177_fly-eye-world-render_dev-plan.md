# fly-eye-world — render 구현 플랜 (P)

## 헤더

- **요구사항**: 1177 — [./20261016-203013-1177_fly-eye-world_req.md](./20261016-203013-1177_fly-eye-world_req.md)
- **인터페이스 계약(I)**: [./20261016-203013-1177_fly-eye-world_interface-design.md](./20261016-203013-1177_fly-eye-world_interface-design.md)(v0.2 · 승인 10-16)
- **모듈**: render
- **작성일**: 2026-10-16
- **Goal**: 헤드리스 크롬 + Babylon 9.25 가 경기장을 싣고, 몸마다 두 눈 큐브맵(6 면 · 작게) + 깊이를 바이너리로 돌려줌. 같은 상태 → 같은 그림.
- **Architecture**: `eyeworld/static.py`(GET 만 · 우리 web + 에코월드 vendor/babylon · assets 읽기 전용 마운트) · `eyeworld/render.py`(크롬 띄우기 · WS 통로 서버 · 메시지 · 시간 초과) · `web/eyeworld/render.html` · `render.js`(장면 · 카메라 · readPixels)
- **Tech Stack**: Babylon.js 9.25.0(에코월드 vendor · 읽기 전용) · 크롬 155(헤드리스 · ANGLE Metal · WebGPU 먼저 · WebGL2 대체) · websockets(선택 묶음 eyeworld)
- **선행 P 의존성**: world(자세 묶음 · 안전)

> 이 저장소의 계획서 형식: Task 마다 파일 · 할 일 · 검증(Run · Expected). 코드는 소스가 정본. 진행 정본은 R 상태.

---

## Task 1: 정적 서버(static)
**Files**: `src/moai_echomind_fly/eyeworld/static.py` · `tests/test_eyeworld_static.py`
- [ ] `ThreadingHTTPServer` · 127.0.0.1 · GET/HEAD 만(그 밖 405) · 마운트 `/web/` → `web/eyeworld/` · `/babylon/` → 에코월드 `vendor/babylon` · `/assets/` → 에코월드 `assets` · 경로 탈출 막음
- Run: `uv run pytest -q tests/test_eyeworld_static.py` · Expected: PUT/POST 405 · `../` 거부 · 금지 포트 거부

## Task 2: 렌더 페이지(web)
**Files**: `web/eyeworld/render.html` · `web/eyeworld/render.js`
- [ ] Babylon 9.25 불러오기(마운트) · 판 확인(`Engine.Version`) · WebGPU 먼저 · 실패 시 WebGL2 · `ready` 보냄
- [ ] `scene.load` — 땅 · 하늘 · 빛 · glb(ImportMeshAsync · 인스턴스) · `scene.ready`(빠진 에셋)
- [ ] `render` — 자세 적용 → 눈마다 정사각 RenderTargetTexture 카메라(90° · face_px) 6 방향 → readPixels → 밝기(초록 가중) u8 + 깊이 f32 → `frames` 바이너리
- Run: 수동 · `scripts/60-eyeworld/v1-render-probe.sh` · Expected: 판 9.25.0 · 6 면 그림 · 알려진 방향의 물체가 맞는 면 · 자리에

## Task 3: 렌더러 몰기(render.py)
**Files**: `src/moai_echomind_fly/eyeworld/render.py` · `tests/test_eyeworld_render_proto.py`
- [ ] 메시지 꼴 `[u32 머리 길이][머리 JSON][본문]` 짜기 · 풀기(시험) · WS 서버(127.0.0.1) · 크롬 띄우기(우리 프로필 · `--headless=new --use-angle=metal --enable-unsafe-webgpu --enable-features=WebGPU` · 페이지 URL) · PID 기록 · 시간 초과 → 한 번 다시 → 다시 띄움 → 두 번 실패면 끝 코드 3
- [ ] 측정: 한 장(눈 둘 × 6 면) 시간 · 크롬 메모리 · face_px 32 · 64 두 크기로 무엇에 비례하나
- Run: `uv run pytest -q tests/test_eyeworld_render_proto.py` · `scripts/60-eyeworld/v1-render-probe.sh` · Expected: 시험 통과 · 같은 상태 두 번 → 같은 그림 해시

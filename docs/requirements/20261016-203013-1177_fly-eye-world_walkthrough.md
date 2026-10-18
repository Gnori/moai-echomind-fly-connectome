# fly-eye-world 검증 보고 — 세계 · 렌더러 · 눈 · 닫힌 고리(보기 창)

- **요구사항**: [R-1177](./20261016-203013-1177_fly-eye-world_req.md) · **설계 문서**: [I v0.3](./20261016-203013-1177_fly-eye-world_interface-design.md) ·
  **계획서**: [world](./20261016-203013-1177_fly-eye-world-world_dev-plan.md) · [render](./20261016-203013-1177_fly-eye-world-render_dev-plan.md) · [eye](./20261016-203013-1177_fly-eye-world-eye_dev-plan.md) · [loop](./20261016-203013-1177_fly-eye-world-loop_dev-plan.md)
- **작성**: 2026-10-18 10:45(`date`) · echomind-fly · **결과**: 조건부 통과(부품 · 보기 창은 됨 · 시각 행동 판정(V1-c · V2)은 가능성 낮아 하지 않음)

> 로그 원문과 측정값만 증거다. 대조군을 이기지 못하면 그대로 적는다(R 2.0-6). 한 보고에 계획서 넷을 묶었다(10-18 요구자 "끝낼 수 있는 크기로" — 이 요구사항을 닫으며).

## 1. 바꾼 파일(10-16 20:30 ~ 10-18 · `git log --name-only`)

| 부분 | 파일 | 무엇 |
|---|---|---|
| world | `src/moai_echomind_fly/eyeworld/{world,scene,safety,static}.py` · `config/eyeworld_scenes/arena-{a,b}.yaml` · `scripts/60-eyeworld/make_arena_b.py` | 세계 서버(원기둥 충돌 · 물체는 안 밀림) · 장면(허가 원본 기록) · 안전(금지 포트 · 에코월드 잠금 · 우리 PID 만) · 정적 서버 |
| render | `src/moai_echomind_fly/eyeworld/render.py` · `web/eyeworld/{render.html,render.js,bind.js}` | 헤드리스 크롬 + Babylon 9.25 · 두 눈 큐브맵 · 에코월드 재질 색인 묶기 · 장면 카메라와 눈 카메라 따로 |
| eye | `src/moai_echomind_fly/eye/{geometry,compound,flow,disk,frontend}.py` | 겹눈 기하 · 낱눈 표본(5° 흐림) · 분석식 광류(V0) · 원판(V0b) · 앞단 v2 · v3b |
| loop | `src/moai_echomind_fly/eyeworld/brainlink.py` · `scripts/60-eyeworld/{view.py,view.sh}` · `web/eyeworld/{view.html,view.js}` · `src/moai_echomind_fly/senses/touch.py` | 보기 창 · 뇌 둘 연결 · 몸 = 하강 뉴런 그대로 · 닿음 신호 · 화면 10-18 다시 짬 |
| 재기 | `scripts/60-eyeworld/{v0_flow,v0b_disk,v0_dose,v1_check,v1_render_probe,v1_scene_probe,f0_frontend}.py` · `scripts/50-analysis/{vision_path,lobula_entry,loom_inputs,v0_graph_checks,touch_entry,touch_probe}.py` | 결과 전 잠금 실행 · 입구 재기 |
| 시험 | `tests/test_eye_{compound,disk,flow,frontend,frontend_v3}.py` · `tests/test_eyeworld_{render_proto,safety,static,world}.py` · `tests/test_senses_touch.py` | 37 개 |

## 2. 실행 로그(원문 · 2026-10-18 10:45:32)

```
$ uv run ruff check src tests
All checks passed!
$ uv run pytest -m "not heavy" -q tests/test_eye_*.py tests/test_eyeworld_*.py tests/test_senses_touch.py
.....................................                                    [100%]
37 passed in 2.89s
$ uv run pytest -m "not heavy" -q
235 passed, 3 deselected, 18 warnings in 10.38s
$ node --check web/eyeworld/view.js && node --check web/eyeworld/render.js && node --check web/eyeworld/bind.js
node --check 셋 통과
```

## 3. 측정값 — 대조군과 나란히

| 단계 | 진짜 배선 | 섞은 배선 · 대조 | 판정 | 근거 |
|---|---|---|---|---|
| V0 회전 → DNa02 | 씨앗 10/10 뒤집힘 · +19~+20 / −15~−17 | 재배선 셋 0 | ✅ C1 | `runs/v0-flow-20261016-205106` |
| V0 다가옴 → LPLC2 | 2.3 | 0 | ❌ 입력 부족 | 같은 곳 |
| V0b 어두운 원판 → LC4 | 210.5 · 같은 쪽 10/10 | 자리 섞기 0~0.2 · 재배선 0 | ✅(다가옴 선택성 없음 — 어두워지기만 200) | `runs/v0b-disk-20261016-214745` |
| 렌더러 | 방향 오차 0.8~2.3° · 한 걸음 10~20 ms · 크롬 1.4 GB · 에코월드 쓰기 0 | — | ✅ 부품 | R-1177 8 장 |
| V1-a · V1-b | 모든 읽기 0 · LC4 7~8 | — | ❌ 입력 부족 → 본 실행 안 함 | `runs/v1-check-20261016-230912` · `-233448` |
| F0(앞단 v3b · 뇌 없음) | F0-1 ✓(맞춤) · F0-4 ✓ 1.00 · F0-3 12.9 · 12.6 Hz | 기준 20 Hz · v2 1.1 Hz | ❌ F0-3 | `runs/f0-frontend-20261017-003141` |
| V0d 뇌 쪽 필요 입력량 | 31.1 Hz 까지 0/10 · 51.8 Hz 6/10 | 재배선 0 | 문턱 > 51.8 Hz → V1-c · V2 안 함 | `runs/v0-dose-20261017-010325` |
| 보기 창(다이얼 1 · 탐색) | 서 있음 · 사람 다가옴 모두 몸 하강 뉴런 0 | 0 | 판정 아님 | `645f875` |

## 4. 정본 문서 동기화

| 고친 정본 | 무엇 |
|---|---|
| 설계 문서 I v0.3 | 3.3c 앞단 v3b · 2.10 보기 창 뇌 연결 · 닿음 · 6 절 촉각 잰 것(이 커밋) |
| `CLAUDE.md` | 저장소 지도(eye · eyeworld · web) · 에코월드 라이브 보호 · 문서 갱신 표 눈 세계 줄(`ccdbf18`) |
| `README.md` 쓰는 법 | 보기 창 절(띄우는 법 · 포트 · 60 분 · 화면 글 그대로의 표 · 정직 줄) — 문서 세션 `fdbd2ce` |
| 연구 노트 · 학습 안내서 | 불필요 — 시각 행동 판정을 하지 않아 결과 절이 없음 · 노트 17 · 18 은 research 가 씀 |

## 5. 남은 것

- 시각 행동 판정(V1-c · V2 F-006 · F-009) — 가능성 낮아 하지 않음(V0d). 다시 하려면 눈 입력을 4 배 넘게 키울 길(장면 · 앞단)이 먼저.
- 텍스처를 넣은 장면에서 F0 다시 재기(기록만 · 앞 F0 는 잎이 회색이었음) · 벤치(`h_` 재질) · 사람 피부 재질.
- 장면 넓히기(F-010 · Could) · 맥 GPU 로 뇌 돌리기(float64 누적 칸 때문에 지금 안 됨).
- 다른 감각 — 감각마다 새 요구사항(촉각 = [R-6031](./20261018-104436-6031_fly-touch_req.md)).

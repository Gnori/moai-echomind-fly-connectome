# fly-npc-staged — 인터페이스 계약 (I) · S0 연결 · S1 반사

- **요구사항**: [R-5138](./20261003-220009-5138_fly-npc-staged_req.md) (G1 확정 2026-10-03)
- **작성일**: 2026-10-03
- **상태**: 초안 — 월드 쪽 `steer` · lockstep 구현(echomind-world) 뒤 엔드포인트 이름을 채워 **G2**
- **원칙**: 지금 초파리 실험 고리(`loop/`)를 고치지 않는다. **몸(FlyGym) 자리에 상자 월드 NPC 를 끼운다** — 뇌(`Brain` · 인코더 · 디코더 · 채널 표)는 그대로 쓴다. 월드 저장소는 고치지 않는다(바깥 HTTP 로만).

---

## 1. 모양

```
결정 하나(모의 0.3 s)마다 — 세계는 멈춰 있다(lockstep)
  ① world.state(npc) + world.neighbors(npc, 반경)   → 좌표 · 방향 · 속도 · 막힘
  ② senses: 좌표 → Obs(지금 인코더가 받는 형식 그대로)          [npc/senses.py]
       · 다가오는 개체 → looming{present, azimuth, elevation, size, expansion}
       · 목표 → odor_sensors 좌 · 우(더듬이 두 개 흉내 · 선언)
       · 막힘 · 가까운 벽 → leg_contact
  ③ 뇌: Loop(body=False) 의 Brain.bin × 결정 구간(지금 코드 그대로) → 하강 뉴런 발화 · 반사 여부
  ④ motor: decoder.drive 의 (왼 · 오른 구동, 반사) → NPC 명령                 [npc/motor.py]
       · 걷기 = 평균 구동 × 최대 속도 · 회전 = (오른 − 왼) × 회전 이득
       · 반사(GF/DNp01 발화) → 다가오는 개체 반대쪽으로 빠르게 물러남(선언)
  ⑤ world.steer(npc, heading, speed, dt=0.3) → world.step(0.3)
  ⑥ 기록 한 줄(decisions.jsonl · 성적표와 같은 칸 + NPC 칸)
```

`Policy`(행동을 정하는 쪽)는 지금 성적표와 같은 자리: S1 은 **고정 걷기**(교사 없음 · 회로 반사만) · S2 는 빠른 정책 · S4 는 층 구조(TieredPolicy · LLM 은 목표만 · 늦은 반영).

---

## 2. 새 모듈 (`src/moai_echomind_fly/npc/`)

| 파일 | 계약 |
|---|---|
| `world_client.py` | `BoxWorld(base_url)` — `state(id)` · `neighbors(id, radius)` · `steer(id, heading_deg, speed, dt)` · `step(dt)` · `add_resident(...)` · `add_object(...)` · `remove_object(ref)` · `navigate_to(id, target)`(S4). **라이브 포트(9212)를 거부**(설정의 금지 포트). 시간 밖 · 막힘은 예외 대신 칸으로 |
| `fake_world.py` | 시험용 가짜 세계(2D 점 · 원 · 직선 이동 · 같은 메서드) — 서버 없이 S0 · S1 고리를 시험 |
| `senses.py` | `to_obs(me, others, goal, cfg) → Obs` — 단위 변환(미터 → 인코더가 기대하는 값) · 각도 부호는 지금 인코더와 같게(+ = 왼쪽) · 확대 속도 = d(시각 크기)/dt(앞 결정 기억) |
| `motor.py` | `to_command(drive, reflex, me, threat, cfg) → (heading_deg, speed)` |
| `scenario.py` | 씨앗 → 장면(NPC 시작 · 목표 위치 · 위험 경로 · 시작 시각) — 같은 씨앗이면 같은 장면 |
| `run.py` | `npc-run --stage S1 --brain real|rewired-sK --seeds …` — 에피소드 고리 · 결과 `runs/npc-<단계>-<시각>/{<구성>.jsonl, <구성>.decisions.jsonl, request.json, card.json}`(성적표와 같은 형식 → 대시보드 재사용) |

설정 `config/npc.yaml`(판 `npc-v1` · 결과 전 선언): 상자 월드 주소 · **금지 포트 [9212]** · 결정 0.3 s · 크기 배율(초파리 mm ↔ NPC m) · 최대 속도 · 회전 이득 · 반사 물러남 속도 · 시간 · 감각 기하(시야 반각 · 사람 눈높이 · 위험 반지름) · 목표 신호(냄새 대신 · 거리 감쇠 · 좌우 더듬이 간격) · 장면 분포 · 에피소드 길이.

---

## 3. 감각 대응(선언 · I 에서 잠금)

| NPC 쪽(좌표) | 초파리 쪽(지금 인코더 입력) | 식 |
|---|---|---|
| 가장 위협적인 다가오는 개체(접근 속도 > 0 · 가까운 것) | `looming.present · azimuth_deg · elevation_deg · size_deg · expansion_deg_s` | 방위 = NPC 머리 방향 기준 각(+ 왼쪽) · 시각 크기 = 2·atan(반지름/거리) · 확대 속도 = 시각 크기 변화 / dt · 고도 = atan((개체 높이 − 눈높이)/거리) |
| 목표(물건 · 장소) | `odor_sensors` 좌 · 우(인코더 `smell.sensors` 자리) | 좌 · 우 감지점(머리 방향 ± 간격)에서 거리 감쇠 농도(선언) — 좌우 차가 방향을 줌 |
| 막힘(blocked) · 가까운 벽 · 사람 | `leg_contact`(다리 기계 감각) | 앞 · 옆 다리에 힘(선언 값) |

## 4. 명령 대응(선언)

| 초파리 쪽 | NPC 명령 |
|---|---|
| `decoder.drive` 왼 · 오른 구동(−0.5~1.5) | 속도 = clip(평균, 0, 1) × 최대 속도 · 방향 변화 = (오른 − 왼) × 회전 이득(도/결정) |
| 반사(도망 뉴런 발화) | 위협 반대 방향으로 물러남 속도(선언) · 결정 하나 동안 다른 명령을 덮음 |
| 멈춤(구동 ≈ 0) | speed 0 |

⚠️ 지금 회로는 **조향 하강 뉴런이 침묵**한다 — S1 은 반사만 보고, S2 는 빠른 정책(2 층)이 걷기 · 돌기를 정한다(policy → 하강 뉴런 목표 발화 → drive · 지금 고리와 같음).

---

## 5. S0 · S1 시험 · 판정(결과 전에 잠금)

- S0(가짜 세계 + 상자 월드): 에피소드 끝까지 돎 · 같은 씨앗 같은 결과(가짜 세계) · 기록 칸 · 라이브 포트 거부 시험.
- S1: 장면 = NPC 가 고정 걷기 · 위험 하나가 정면 ± 60° 에서 다가옴. 구성 = `real` · `rewired-s0/s1/s2` · `no_brain`(반사 없음 · 그대로 걸음). 지표 = 부딪힘 비율(거리 < 위험 반지름 + NPC 반지름). 판정 = 진짜 < 섞은 배선 전부 · 씨앗 부호 뒤집기 p < 0.05(성적표 scorecard-v2 와 같은 통계) · `no_brain` 보다 낮음.

## 6. 미정 — echomind-world 구현 뒤 채움

- `steer` · `step` · 주변 개체 입구의 실제 경로 · 응답 칸 이름 — heading 은 **세계 기준**(echomind-world 10-03) · 세 입구 모두 설정 스위치 뒤 기본 꺼짐 · 상자 월드에서만(브랜치 fly-steer-1003)
- 상자 월드 띄우는 명령 · 포트
- lockstep 이 안 되면: 시간 배율로 얼마나 느려지는지(S0 에서 잼) — 안 되면 S1 을 실시간보다 느린 대신 "결정 사이 세계 진행"의 오차를 기록

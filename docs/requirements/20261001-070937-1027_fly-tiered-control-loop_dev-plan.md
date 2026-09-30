# fly-tiered-control — loop 구현 플랜 (P)

## 헤더

- **요구사항**: 1027 — [./20261001-070937-1027_fly-tiered-control_req.md](./20261001-070937-1027_fly-tiered-control_req.md)
- **인터페이스 계약(I)**: 생략 — R 3.3 축약(한 모듈 loop + 새 설정 파일 하나)
- **모듈**: loop — 새 `loop/tiered.py` · `loop/fast_policy.py` · 기존 `loop/scorecard.py` · `cli.py` · 새 `config/control.yaml` · 새 시험
- **작성일**: 2026-10-01
- **G1**: 요구자 2026-10-01 *"개발 먼저 해야할 건들 미리 개발을 해놔 검토 잘 하고 영향도 파악 잘해서"* · *"물론 설계를 잘 해야겠지만"* → 2장 기능 확정 · **문턱 숫자는 F-007 조사 뒤 잠금(그 전엔 설정에 "미정")**.
- **Goal**: Qwen 을 **사건 때만** 부르는 층 구조를, 지금 결과를 하나도 바꾸지 않는 방식(끼우는 감싸개)으로 만든다.

---

## 1. 설계

### 1.1 끼우는 자리 — 루프는 고치지 않는다

`Loop.episode` 는 결정마다 `policy(vectors, truth, obs, d) → (action, info)` 만 부른다. 층 구조는 **이 policy 를 감싸는 `TieredPolicy`** 하나다.
→ 몸 · 뇌 · 냄새 맡기 구간(`sniff_decisions` 는 policy 뒤에서 STOP 으로 덮음) · 도망 반사(`decoder.drive` 가 뇌 구간 안에서) 는 **그대로**.

```
TieredPolicy.__call__(vectors, truth, obs, d):
    z  = features(vectors[primary])              # 투영기와 같은 압축 입력(뇌마다 자기 압축기)
    r  = reflex.check(z)                          # 1층(F-005) — 걸리면 그 행동 · tier="reflex"
    f  = fast.predict(z) → (action, confidence, heads)   # 2층(F-002)
    tr = gate.trigger(z, f, d)                    # F-003 — None 이면 부르지 않음
    if tr: a = llm(vectors, truth, obs, d) · tier="llm" · trigger=tr · gate.remember(z, d)
    else:  a = f.action(또는 hold 모드면 마지막 LLM 행동) · tier="fast"
    info = {..., "tier", "trigger", "fast_conf", "llm_calls"}
```

### 1.2 정직 규칙 — 참값을 쓰지 않는다

- 게이트 · 빠른 정책 · 반사 규칙은 **초파리 신호(압축 특징)만** 본다. `truth` 는 넘겨받아도 쓰지 않는다(시험으로 막음: truth 를 망가뜨려도 층 결정이 같아야 함).
- 교사 행동은 **빠른 정책 학습 정답**으로만(자료의 `teacher_action`).
- 진짜 배선 · 재배선 **각자 자기 압축기 · 자기 빠른 정책** — 같은 절차(투영기와 같은 원칙).

### 1.3 2층 빠른 정책(F-002)

- 입력: 자료 `fly-loop-v4` 학습 나눔의 채널 특징 → `compress_for`(투영기와 같은 압축 · 관문 통과 채널: 시각 · smell_ln · 공간).
- 정답: 교사 행동 → **돌기 3 갈래**(left · none · right) · **걷기 2 갈래**(walk>0 · 멈춤) · **위험 머리**(교사 돌기가 위협 회피였나 — 참값이 아니라 교사 규칙의 분기 이름을 학습 정답으로만).
- 방법: 능선 회귀(기존 `linear_probe.ProbeSet` 과 같은 λ 교차검증) → 선형 가중치를 파일로(`runs/fast-policy-<뇌>-<판>/weights.npz` · meta).
- 확신도: 두 높은 점수 차(여유 폭). 추론 = 행렬 곱 한 번(목표 < 5 ms · 실측).
- 행동 조립: 돌기 방향 · 세기 = 교사 규칙의 이산값 · 걷기 = 0.8/0.6(교사와 같은 값).

### 1.4 호출 조건(F-003) — 계산만 먼저, 문턱은 조사 뒤 잠금

| 조건 | 계산 | 문턱(control.yaml) |
|---|---|---|
| 첫 결정 · 냄새 맡기 끝 | `d == sniff_decisions`(목표 설정) | 고정 |
| 시간 상한 | 마지막 호출 뒤 결정 수 ≥ N | `max_gap_decisions` |
| 놀람 | ‖z_t − z_{t−1}‖ / 학습 나눔 같은 양의 분위수 | `surprise_q` |
| 애매함 | 빠른 정책 돌기 확신도 < c | `confidence_min` |
| 충돌 | 위험 머리 점수 > t **그리고** 먹이 쪽 돌기 확신도 > t | `conflict_min` |

모드 `always`(늘 부름 = 지금 동작) · `never`(fast-only) · `gated`. 문턱이 "미정"이면 `gated` 는 시작을 거부(결과 전에 잠그게 강제).

### 1.5 성적표(F-004)

- 새 구성 이름 규칙: `<기존>-gated`(vector-gated · rewired-sK-gated) · `fast-only`(진짜 · LLM 없음) · `rewired-sK-fast-only`.
- `scorecard-v4` 기록: decisions.jsonl 에 `tier` · `trigger` · `fast_conf` · 에피소드 요약에 `llm_calls` · `call_rate` · `est_wall_s`(= 호출 × 실측 지연 + 결정 × 빠른 정책 시간).
- 판정(R 2.4 성공 기준): 기존 scorecard-v2 판정에 **gated 쌍**(vector-gated 대 rewired-sK-gated)을 더하고 호출 비율 · 벽시계 비율을 곁들인다. 기존 구성의 결과 · 판정은 **바뀌지 않는다**.

---

## 2. 영향도

| 대상 | 변경 | 영향 | 막는 법 |
|---|---|---|---|
| `loop/episode.py` | **없음** | — | 감싸개로 끼움 |
| `loop/scorecard.py` | 구성 이름 해석 · TieredPolicy 조립 · 기록 칸 · 요약 칸 | 기존 구성의 결과 · 판정이 달라질 위험 | 회귀 시험: 기존 구성은 코드 경로가 그대로(새 칸은 없거나 기본값) · `always` 모드 = 기존과 같은 행동 |
| decisions.jsonl · 에피소드 요약 형식 | 칸 추가(scorecard-v4) | 대시보드 episodes 쪽 · decision_turns.py 가 새 칸을 모름 | 칸 **추가만**(삭제 · 이름 변경 없음) · 없으면 무시하도록 읽는 쪽 확인 |
| 새 `config/control.yaml` | 추가 | 없음(아무것도 아직 안 읽음) | — |
| 기존 설정(task · channels · …) | **없음** | 지금 도는 v7 성적표가 구성마다 다시 읽는다 | 성적표 끝날 때까지 손대지 않음 |
| `cli.py` | `fast-policy fit/eval` 명령 등록 | 없음 | — |
| 빠른 정책 학습(CPU) | 새 폴더 `runs/fast-policy-*` | CPU 수 분 · GPU 안 씀 · 잠금 불필요(가드는 무거운 실행만) | 성적표와 CPU 를 조금 나눠 쓸 뿐 — 실측 기록 |
| 학습 코드(train_cli · projector) | **없음** | — | — |

---

## Task 1: 설정 · 틀 (F-001 · F-003)
- [ ] `config/control.yaml`(control-v1 · mode · 문턱 "미정" · between_calls fast|hold)
- [ ] `loop/tiered.py` — `Gate`(조건 계산 · 기억) · `TieredPolicy`(1.1) · 참값을 안 씀
- [ ] 시험: always = 매 결정 LLM · never = LLM 0 · 시간 상한 · 놀람 · 애매함 · 충돌 각각 가짜 특징으로 · truth 를 망가뜨려도 결정 같음 · 문턱 미정이면 gated 거부

## Task 2: 빠른 정책 (F-002)
- [ ] `loop/fast_policy.py` — fit(자료 · 뇌 · 압축) · predict(z → 행동 · 확신도 · 머리) · 저장/불러오기 · `fast-policy fit|eval` 명령
- [ ] 시험: 심은 선형 규칙을 되찾음 · 저장 → 불러오기 같음 · 추론 < 5 ms
- [ ] 실행(CPU): 진짜 · 재배선 s0~s2 fit + 평가 나눔 일치율 · 확신도 보정 → R 2.2 인수 기준 기록

## Task 3: 반사 규칙 (F-005)
- [ ] 회로 신호로 읽을 수 있는 것만 — 빠른 정책 위험 머리 강함 → 회피 · (미각 채널이 관문 밖이라) 먹이 닿음은 "회로 신호 없음"으로 기록 · 교사와 다른 반사 비율

## Task 4: 성적표 (F-004)
- [ ] 구성 이름 해석 · 조립 · scorecard-v4 칸 · 요약 · 판정에 gated 쌍
- [ ] 회귀 시험: 기존 구성 목록이면 기록 칸 · 판정이 v3 과 같음(새 칸 제외)
- [ ] 대시보드 episodes 쪽 층 표시 — doc 세션(칸 추가만 알림)
- [ ] 실행(무거움): A · B 뒤 · 문턱 잠근 뒤 · 요구자 확인

## 최종 검증
- `uv run ruff check src tests scripts` · `uv run pytest -m "not heavy"` — 파이프 없이 끝 코드 확인 뒤 커밋

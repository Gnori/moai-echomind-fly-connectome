# fly-connectome-closed-loop — loop 검증 보고 (W)

## 메타데이터

- **요구사항**: 3340 — [./20260921-233859-3340_fly-connectome-closed-loop_req.md](./20260921-233859-3340_fly-connectome-closed-loop_req.md)
- **대응 P**: [./20260921-233859-3340_fly-connectome-closed-loop-loop_dev-plan.md](./20260921-233859-3340_fly-connectome-closed-loop-loop_dev-plan.md)
- **관련 I**: [./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md](./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md)
- **작성일**: 2026-09-23 05:0x
- **결과**: **조건부 통과** — 닫힌 고리와 성적표는 요구대로 나왔다(F-010 ~ F-017 구현·실행 완료).
  다만 **`vector` 가 대조군 `rewired` 를 이기지 못했다** → R 2.0-6 의 문구 규칙 발동:
  이 결과로 "초파리 뇌 덕분"이라고 쓰지 않는다.

---

## 1. 변경 파일 목록

저장소 `GitHub/moai-echomind-fly-connectome` (형제 저장소 · moai-echomind 아님).

| 파일 | 변경 유형 | 요약 |
|---|---|---|
| `body/fly_body_server.py` · `body/pyproject.toml` | 추가 | 몸 = FlyGym 1.x 별도 3.12 프로세스(표준 입출력 JSON 한 줄) |
| `src/moai_echomind_fly/loop/{world,encoder,signals,decoder,episode,teacher,collect}.py` | 추가 | 세계·감각→자극·채널 벡터 9종·행동 디코더·조율기·교사·자료 모으기 |
| `src/moai_echomind_fly/loop/{qwen,fly_prompt,projector}.py` | 추가 | Qwen3.8-27B 직접 로드(얼림) · `<|vision_pad|>` 자리 · 채널별 투영기 |
| `src/moai_echomind_fly/loop/{train,train_cli}.py` | 추가·수정 | 투영기 학습(층별 역전파) · 캡션 평가 · 이어 가기(train-v2) |
| `src/moai_echomind_fly/loop/scorecard.py` | 추가 | 다섯 구성 성적표 |
| `src/moai_echomind_fly/loop/linear_probe.py` | 추가 | Qwen 없이 벡터만으로 칸 맞히기(원인 가르기) |
| `src/moai_echomind_fly/{datasets,guard}.py` | 추가·수정 | 학습 자료 묶음 · 실행 중 메모리 감시(Watchdog) |
| `config/{channels,task,llm,guard}.yaml` · `bin/run_full.sh` | 추가·수정 | 채널 표·과제·가중치 경로·문턱 · 전체 순서 스크립트 |
| `tests/{test_loop_parts,test_train_layerwise}.py` | 추가 | 교사·디코더·채널·선형 검사 · 층별 역전파 기울기 동일성 |

커밋: `7fcff9b`(부품) · `4ac4635`(학습·평가·성적표) · `7c0abf0`(자료 묶음) · `fbd1690`·`b3e87c2`(캐시·감시) ·
`59b9b29`(층별 역전파) · `7818a7b`(수치 정정) · `91bdf67`(이어 가기 · train-v2) · `d68bba7`(선형 검사).

---

## 2. 테스트 실행 로그 (원문)

```
$ uv run pytest -q -m "not heavy"
40 passed, 3 deselected, 18 warnings in 4.01s

$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q tests/test_train_layerwise.py
1 passed in 1.66s
```

메모리 측정 `runs/train-probe-20260922-203433.jsonl` (12 걸음 · 고침 뒤):

```
{"step": 1, "task": "act", "sec": 22.25, "loss": 0.645, "active_gb": 29.1, "cache_gb": 0.15, "peak_gb": 32.21}
{"step": 12, "task": "cap", "sec": 14.71, "loss": 3.234, "active_gb": 29.0, "cache_gb": 0.22, "peak_gb": 31.95}
```

(고치기 전 `runs/train-probe-20260922-200949.jsonl`: `cache_gb` 74.7 → 76.3 · `peak_gb` 101.8 · 92~314 s/걸음)

---

## 3. 모듈 동작 확인 기록

### 3.1 전체 순서 실행 — `runs/full-20260922-220714.log`

```
=== 2026-09-22 22:07:14 여유 86% · train --brain real --resume-caption runs/projector-real-train-v1/stage1-caption.safetensors
=== 2026-09-22 23:37:44 끝 코드 0 · train --brain real ...
=== 2026-09-23 02:52:53 끝 코드 0 · train --brain rewired
=== 2026-09-23 02:58:43 / 03:04:26 / 03:10:08 / 03:15:57 끝 코드 0 · eval-caption ×4
=== 2026-09-23 04:49:51 끝 코드 0 · scorecard --seeds 2000-2009
=== 2026-09-23 04:49:51 전체 끝
```

### 3.2 학습 (F-017) — `runs/projector-{real,rewired}-train-v2/`

| | 진짜 배선 | 섞은 배선 |
|---|---|---|
| 학습 전 검증(캡션 / 행동) | 3.36 / 0.41 | 3.380 / 0.408 |
| 캡션 단계 끝 | 0.106 / 0.325 | 0.107 / 0.359 |
| 행동 단계 끝(최종) | 0.132 / **0.111** | 0.154 / **0.087** |
| Qwen 가중치 지문 | `a4cfcb8d…` 학습 전후 **같음** | 같음 |

- 규칙 `train-v2`(행동 단계는 새 옵티마이저) · 진짜 배선은 v1 캡션 저장본(sha256 `30d221dc…`)에서 이어 감.
- 걸음당 5.6~7.6 s · 최대 32.5 GB · 12 걸음 측정과 같은 수준으로 평탄.

### 3.3 캡션 평가 (F-017) — 학습에 안 쓴 씨앗 105 결정

| 칸 | 진짜-캡션 | 섞은-캡션 | 진짜-최종 | 섞은-최종 | 최빈값 찍기 |
|---|---|---|---|---|---|
| loom_present | 0.686 | 0.667 | 0.676 | 0.752 | 0.667 |
| loom_side | 0.676 | 0.667 | 0.676 | 0.686 | 0.667 |
| odor_side | 0.533 | 0.543 | 0.238 | 0.238 | 0.543 |
| legs | 0.438 | 0.438 | 0.324 | 0.333 | 0.333 |

→ **네 경우 모두 찍기 수준**이고 두 뇌가 구별되지 않는다. 행동 단계를 거치면 캡션이 오히려 나빠진다.

### 3.4 성적표 (F-016) — `runs/scorecard-20260923-031557/scorecard.json` · 씨앗 2000~2009

| 구성 | 먹이 도착 | 끝 거리(mm) | 회피 | 형식 위반 | 결정 수 | 결정 1회(s) |
|---|---|---|---|---|---|---|
| vector | **0.2** | 41.24 | 1.0 | 0 | 177 | 1.91 |
| text | 0.0 | 52.79 | 1.0 | 0 | 200 | 3.04 |
| no_circuit | **0.9** | 4.75 | 0.75 | 0 | 111 | 1.91 |
| rewired | **0.2** | 38.59 | 1.0 | 0 | 179 | 1.95 |
| no_llm | 0.0 | 26.22 | 0.0 | 0 | 200 | 0.0 |

```json
"vector_beats_rewired": false,
"wording": "섞은 배선 대조를 이기지 못했다 — '초파리 뇌'라는 말을 쓰지 않는다"
```

⚠️ 회피율은 가르는 힘이 없다 — 가만히 있는 `no_llm` 만 0.0 이고 움직이는 구성은 모두 1.0 이다.

### 3.5 원인 가르기 — `runs/linear-probe-20260923-031800.json` (Qwen 없이 · CPU 14 s)

전 채널(3.2만 차원)을 한꺼번에: 두 뇌 모두 찍기 수준(진짜 0.714 / 0.600 / 0.543 · 섞은 0.733 / 0.610 / 0.419).

칸에 맞는 채널만:

| 채널 → 칸 | 진짜 배선 | 섞은 배선 | 찍기 |
|---|---|---|---|
| 시각 → loom_present | **0.810** | 0.743 | 0.667 |
| 시각 → loom_side | **0.743** | 0.590 | 0.667 |
| 몸 → legs | **0.590** | 0.419 | 0.333 |
| 촉각 → legs | 0.476 | 0.448 | 0.333 |
| 후각 → odor_side | 0.457 | 0.467 | 0.543 |

→ **신호에는 정보가 있고(시각·몸) 진짜 배선이 섞은 배선보다 높다.** 지금 막힌 곳은 앞(회로)이 아니라
**전달(3.2만 차원을 한꺼번에 · 투영기 3,190만 매개변수 대 표본 581)** 이다.
`no_circuit` 0.9 도 같은 쪽을 가리킨다 — Qwen 은 값을 받으면 과제를 푼다.

---

## 4. 정본 문서 동기화 내역

| 갱신한 정본 문서 | 갱신 내용 |
|---|---|
| 연구 노트 11장 — 형제 저장소 `moai-echomind-fly-connectome/docs/research/fly-connectome/` `11-구현-초파리-신호를-Qwen-에.md`(2026-09-23 옮김) | 8장에 멈춤·고침·재개, 결과 장(8.5) 추가(요구자용 설명 · HTML 같이 빌드) |
| R-3340 §1.3-10 · 1.3-11 | 학습 메모리(0.32 GB/토큰 → 층별 역전파 32 GB) · train-v2 |
| [echomind-fly-connectome-handoff.md](./echomind-fly-connectome-handoff.md) | 결과·원인 가르기·다음 판단거리 |

---

## 5. 잔여 이슈

1. **`vector` 가 `rewired` 를 못 이긴다**(둘 다 0.2) — R 2.0-6 문구 규칙 적용 중. 다음 시도 후보:
   ① 채널·차원 줄여 넣기(PCA 등) ② 캡션 손실이 **틀 글자**에 쏠리는 문제(칸 글자에 가중) ③ 자료 581 → 더 많이
   ④ 투영기 크기(3,190만)를 표본에 맞게 줄이기. **요구자와 정한 뒤** 진행.
2. **후각 → 냄새 방향을 두 뇌 모두 못 읽는다**(0.457 / 0.467 · 찍기 0.543) — 냄새 좌우 표현이 벡터에
   남는지 따로 봐야 한다(자극 뉴런 제외와 관련 가능).
3. **행동 단계가 캡션 능력을 깎는다**(진짜 0.106 → 0.132, 평가로는 0.533 → 0.238) — 캡션 섞기 비율 0.3 재검토.
4. F-006(A 단계 Could)과 B 단계 "클로드 코드 자체 연결"은 그대로 남아 있다.
5. 학습 재현성: 투영기 초기값에 씨앗을 주지 않는다(같은 설정이라도 정확히 같은 결과는 안 나온다).
   → **2026-09-23 고침**(요구자 "남은거 해바"): 규칙 판 `train-v3` = v2 + `init_seed` · 끝난 결과 폴더 덮어쓰기 막음 ·
   학습 · 평가가 묶음 `data/datasets/fly-loop-v1` 을 직접 읽음(조각과 전부 같음 대조) · 지금 결과는 `--version train-v2` 로 다시 평가.
   끝 저장본의 비트 단위 재현은 아직 재 보지 않았다(GPU 계산 순서). 시험 42 통과(`test_projector_init_seed_makes_same_weights` 등).
6. (3.5 보충 · 2026-09-23) 채널별 선형 검사가 이제 파일로 남는다 — `runs/linear-probe-20260923-094302.json` · **위 3.5 표와 같은 값**.
   새로 보인 것: 시각 → 물체 거리 0.771 / 0.581 · 움직임 0.781 / 0.571(진짜 / 섞은). 운동 → 냄새 쪽 0.619 / 0.457 은
   하강 뉴런이 교사 명령을 받으므로 정답이 새어 든 것일 수 있어 근거로 쓰지 않는다.
